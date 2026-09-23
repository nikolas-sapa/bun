use core::marker::PhantomData;
use std::borrow::Cow;

use crate::Error;
use crate::bun_fs as fs;
use bun_alloc::AstAlloc;
use bun_ast::{ImportKind, ImportRecord, ImportRecordFlags, ImportRecordTag, Index as AstIndex};
use bun_ast::{Loc, Log, Range, Source};
use bun_core::strings;
use bun_paths::fs::Path as FsPath;
use bun_paths::{platform, resolve_path};
use bun_sys as sys;

/// The single `lol_html::OutputSink` type every Bun `HtmlRewriter` is built
/// with (here and in `HTMLRewriter`), so lol_html's rewriter/tokenizer
/// machinery is instantiated once rather than once per sink closure type.
pub enum OutputSink<'a> {
    /// `HTMLRewriter`'s response pipe: every chunk is appended to its staging
    /// buffer (statically dispatched — this is the throughput-sensitive user).
    /// The pipe owns the rewriter that owns this sink, so the back-reference
    /// invariant holds structurally.
    Buffer(bun_ptr::BackRef<bun_ptr::JsCell<Vec<u8>>>),
    /// The bundler's HTML scanner.
    Callback(Box<dyn FnMut(&[u8]) + 'a>),
}

impl lol_html::OutputSink for OutputSink<'_> {
    #[inline]
    fn handle_chunk(&mut self, chunk: &[u8]) {
        match self {
            // lol-html signals end-of-document with one zero-length chunk.
            OutputSink::Buffer(buffer) => {
                if !chunk.is_empty() {
                    buffer.with_mut(|buffer| buffer.extend_from_slice(chunk));
                }
            }
            OutputSink::Callback(callback) => Self::call(callback, chunk),
        }
    }
}

impl OutputSink<'_> {
    // Out of line so the per-chunk sink call lol_html inlines everywhere stays small.
    #[cold]
    #[inline(never)]
    fn call(callback: &mut (dyn FnMut(&[u8]) + '_), chunk: &[u8]) {
        callback(chunk)
    }
}
use lol_html::html_content::Element;

bun_core::declare_scope!(HTMLScanner, hidden);

pub(crate) struct HTMLScanner<'a> {
    // arena field dropped — global mimalloc (see PORTING.md §Allocators).
    pub import_records: Vec<ImportRecord>,
    pub log: &'a mut Log,
    pub source: &'a Source,
}

impl<'a> HTMLScanner<'a> {
    pub(crate) fn init(log: &'a mut Log, source: &'a Source) -> HTMLScanner<'a> {
        HTMLScanner {
            import_records: Vec::new(),
            log,
            source,
        }
    }
}

/// The character reference at the start of `text` and its length. Like the
/// HTML parser, a number that names no usable character reads as U+FFFD.
fn char_ref(text: &[u8]) -> Option<(char, usize)> {
    const NAMED: [(&[u8], char); 5] = [
        (b"&amp;", '&'),
        (b"&lt;", '<'),
        (b"&gt;", '>'),
        (b"&quot;", '"'),
        (b"&apos;", '\''),
    ];
    if let Some((name, c)) = NAMED.iter().find(|(name, _)| text.starts_with(name)) {
        return Some((*c, name.len()));
    }
    let (radix, digits_at) = match text {
        [b'&', b'#', b'x' | b'X', ..] => (16, 3),
        [b'&', b'#', ..] => (10, 2),
        _ => return None,
    };
    let mut code_point = Some(0u32);
    let mut end = digits_at;
    while let Some(digit) = text.get(end).and_then(|&d| char::from(d).to_digit(radix)) {
        code_point = code_point.and_then(|value| value.checked_mul(radix)?.checked_add(digit));
        end += 1;
    }
    if end == digits_at || text.get(end) != Some(&b';') {
        return None;
    }
    let c = code_point
        .and_then(char::from_u32)
        .filter(|c| !c.is_control())
        .unwrap_or(char::REPLACEMENT_CHARACTER);
    Some((c, end + 1))
}

/// The URL that an attribute value spells (`&amp;` is how HTML writes `&`).
fn decode_char_refs(value: &[u8]) -> Cow<'_, [u8]> {
    let Some(mut ampersand) = strings::index_of_char_usize(value, b'&') else {
        return Cow::Borrowed(value);
    };
    let mut url = Vec::with_capacity(value.len());
    let mut rest = value;
    loop {
        url.extend_from_slice(&rest[..ampersand]);
        rest = &rest[ampersand..];
        let (c, len) = char_ref(rest).unwrap_or(('&', 1));
        url.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
        rest = &rest[len..];
        match strings::index_of_char_usize(rest, b'&') {
            Some(next) => ampersand = next,
            None => {
                url.extend_from_slice(rest);
                return Cow::Owned(url);
            }
        }
    }
}

/// `scheme:...` or `//host/...`. The resolver marks it external as written.
fn url_is_remote(url: &[u8]) -> bool {
    if url.starts_with(b"//") {
        return true;
    }
    let mut len = 0;
    while len < url.len()
        && (url[len].is_ascii_alphanumeric() || matches!(url[len], b'+' | b'-' | b'.'))
    {
        len += 1;
    }
    len > 0 && url[0].is_ascii_alphabetic() && url.get(len) == Some(&b':')
}

/// Where the `?query#fragment` starts. `#top` alone has no path before it.
fn url_suffix_index(url: &[u8]) -> usize {
    match strings::index_of_any(url, b"?#") {
        Some(0) | None => url.len(),
        Some(index) => index,
    }
}

/// The `?query#fragment` of a local `src`/`href` value. Empty for a remote URL.
pub(crate) fn url_suffix(value: &[u8]) -> Cow<'_, [u8]> {
    let url = decode_char_refs(value);
    if url_is_remote(&url) {
        return Cow::Borrowed(b"");
    }
    let index = url_suffix_index(&url);
    match url {
        Cow::Borrowed(url) => Cow::Borrowed(&url[index..]),
        Cow::Owned(mut url) => {
            url.drain(..index);
            Cow::Owned(url)
        }
    }
}

/// The file name that a URL path spells. `None` keeps the path as written: a
/// malformed escape (`%PUBLIC_URL%`), bytes that are not UTF-8 (`%E9`), or a
/// byte that the output URL, which is the raw file name, cannot carry as itself.
fn percent_decode(path: &[u8]) -> Option<Vec<u8>> {
    let mut escape = strings::index_of_char_usize(path, b'%')?;
    let mut decoded = Vec::with_capacity(path.len());
    let mut rest = path;
    loop {
        let byte = bun_core::fmt::hex_pair_value(*rest.get(escape + 1)?, *rest.get(escape + 2)?)?;
        if byte.is_ascii_control() || matches!(byte, b'#' | b'?' | b'%' | b'/' | b'\\') {
            return None;
        }
        decoded.extend_from_slice(&rest[..escape]);
        decoded.push(byte);
        rest = &rest[escape + 3..];
        match strings::index_of_char_usize(rest, b'%') {
            Some(next) => escape = next,
            None => break,
        }
    }
    decoded.extend_from_slice(rest);
    strings::is_valid_utf8(&decoded).then_some(decoded)
}

impl<'a> HTMLScanner<'a> {
    fn create_import_record(
        &mut self,
        value: &[u8],
        url_attribute: &[u8],
        kind: ImportKind,
    ) -> Result<(), Error> {
        // A `srcset` is a list of URLs, not a URL. It is resolved as written.
        let is_url = url_attribute != b"srcset";
        let url = if is_url {
            decode_char_refs(value)
        } else {
            Cow::Borrowed(value)
        };
        let is_remote = url_is_remote(&url);
        let decoded;
        let input_path: &[u8] = if is_remote || !is_url {
            value
        } else {
            let path = &url[..url_suffix_index(&url)];
            decoded = percent_decode(path);
            decoded.as_deref().unwrap_or(path)
        };
        // In HTML, sometimes people do /src/index.js
        // In that case, we don't want to use the absolute filesystem path, we want to use the path relative to the project root
        let path_to_use: &[u8] = if is_remote {
            input_path
        } else if input_path.len() > 1 && input_path[0] == b'/' {
            resolve_path::join_abs_string::<platform::Auto>(
                fs::FileSystem::instance().top_level_dir,
                &[&input_path[1..]],
            )
        }
        // Check if imports to (e.g) "App.tsx" are actually relative imoprts w/o the "./"
        else if input_path.len() > 2 && input_path[0] != b'.' && input_path[1] != b'/' {
            'blk: {
                let Some(index_of_dot) = bun_core::strings::last_index_of_char(input_path, b'.')
                else {
                    break 'blk input_path;
                };
                let ext = &input_path[index_of_dot..];
                if ext.len() > 4 {
                    break 'blk input_path;
                }
                // /foo/bar/index.html -> /foo/bar
                let dirname = resolve_path::dirname::<platform::Auto>(self.source.path.text());
                if dirname.is_empty() {
                    break 'blk input_path;
                }
                let resolved =
                    resolve_path::join_abs_string_z::<platform::Auto>(dirname, &[input_path]);
                if sys::exists_z(resolved) {
                    resolved.as_bytes()
                } else {
                    input_path
                }
            }
        } else {
            input_path
        };

        let owned: &'static [u8] =
            Box::leak(AstAlloc::vec_from_slice(path_to_use).into_boxed_slice());
        let record = ImportRecord {
            path: FsPath::init(owned),
            kind,
            range: Range::NONE,
            tag: ImportRecordTag::default(),
            loader: None,
            source_index: AstIndex::default(),
            original_path: b"",
            flags: ImportRecordFlags::default(),
        };

        self.import_records.push(record);
        Ok(())
    }

    fn on_write_html(&mut self, bytes: &[u8]) {
        let _ = bytes; // bytes are not written in scan phase
    }

    fn on_html_parse_error(&mut self, message: &[u8]) {
        // Vec/Box allocations abort on OOM; just call. `IntoText for
        // Vec<u8>` → `Cow::Owned`, so the Log owns and drops the copy.
        let _ = self
            .log
            .add_error(Some(self.source), Loc::EMPTY, message.to_vec());
    }

    fn on_tag(
        &mut self,
        _element: &mut Element<'_, '_>,
        path: &[u8],
        url_attribute: &[u8],
        kind: ImportKind,
    ) {
        let _ = self.create_import_record(path, url_attribute, kind);
    }

    pub(crate) fn scan(&mut self, input: &[u8]) -> Result<(), Error> {
        Processor::run(self, input)
    }
}

type Processor<'a> = HTMLProcessor<HTMLScanner<'a>, false>;

// ───────────────────────────────────────────────────────────────────────────
// HTMLProcessor — generic over visitor `T` and `VISIT_DOCUMENT_TAGS`
// ───────────────────────────────────────────────────────────────────────────

/// Trait capturing the methods `HTMLProcessor` calls on `T`.
pub(crate) trait HTMLProcessorHandler {
    fn on_tag(
        &mut self,
        element: &mut Element<'_, '_>,
        path: &[u8],
        url_attribute: &[u8],
        kind: ImportKind,
    );
    fn on_write_html(&mut self, bytes: &[u8]);
    fn on_html_parse_error(&mut self, message: &[u8]);

    // Only required when VISIT_DOCUMENT_TAGS == true; `run` only calls
    // these when visiting document tags, so the defaults are never
    // reached for handlers that don't visit document tags.
    fn on_body_tag(&mut self, _element: &mut Element<'_, '_>) -> bool {
        unreachable!()
    }
    fn on_head_tag(&mut self, _element: &mut Element<'_, '_>) -> bool {
        unreachable!()
    }
    fn on_html_tag(&mut self, _element: &mut Element<'_, '_>) -> bool {
        unreachable!()
    }
}

impl<'a> HTMLProcessorHandler for HTMLScanner<'a> {
    fn on_tag(
        &mut self,
        element: &mut Element<'_, '_>,
        path: &[u8],
        url_attribute: &[u8],
        kind: ImportKind,
    ) {
        HTMLScanner::on_tag(self, element, path, url_attribute, kind)
    }
    fn on_write_html(&mut self, bytes: &[u8]) {
        HTMLScanner::on_write_html(self, bytes)
    }
    fn on_html_parse_error(&mut self, message: &[u8]) {
        HTMLScanner::on_html_parse_error(self, message)
    }
}

pub(crate) struct HTMLProcessor<T, const VISIT_DOCUMENT_TAGS: bool>(PhantomData<T>);

#[derive(Clone, Copy)]
struct TagHandler {
    /// CSS selector to match elements
    pub(crate) selector: &'static str,
    /// The attribute to extract the URL from
    pub(crate) url_attribute: &'static str,
    /// The kind of import to create
    pub(crate) kind: ImportKind,
}

impl TagHandler {
    const fn new(selector: &'static str, url_attribute: &'static str, kind: ImportKind) -> Self {
        Self {
            selector,
            url_attribute,
            kind,
        }
    }
}

const TAG_HANDLERS: [TagHandler; 16] = [
    // Module scripts with src
    TagHandler::new("script[src]", "src", ImportKind::Stmt),
    // CSS Stylesheets
    TagHandler::new("link[rel='stylesheet'][href]", "href", ImportKind::At),
    // CSS Assets
    TagHandler::new("link[as='style'][href]", "href", ImportKind::At),
    // Font files
    TagHandler::new(
        "link[as='font'][href], link[type^='font/'][href]",
        "href",
        ImportKind::Url,
    ),
    // Image assets
    TagHandler::new("link[as='image'][href]", "href", ImportKind::Url),
    // Audio/Video assets
    TagHandler::new(
        "link[as='video'][href], link[as='audio'][href]",
        "href",
        ImportKind::Url,
    ),
    // Web Workers
    TagHandler::new("link[as='worker'][href]", "href", ImportKind::Stmt),
    // Manifest files
    TagHandler::new("link[rel='manifest'][href]", "href", ImportKind::Url),
    // Icons
    TagHandler::new(
        "link[rel='icon'][href], link[rel='apple-touch-icon'][href]",
        "href",
        ImportKind::Url,
    ),
    // Images with src
    TagHandler::new("img[src]", "src", ImportKind::Url),
    // Images with srcset
    TagHandler::new("img[srcset]", "srcset", ImportKind::Url),
    // Videos with src
    TagHandler::new("video[src]", "src", ImportKind::Url),
    // Videos with poster
    TagHandler::new("video[poster]", "poster", ImportKind::Url),
    // Audio with src
    TagHandler::new("audio[src]", "src", ImportKind::Url),
    // Source elements with src
    TagHandler::new("source[src]", "src", ImportKind::Url),
    // Source elements with srcset
    TagHandler::new("source[srcset]", "srcset", ImportKind::Url),
    //     // Iframes
    //     TagHandler::new("iframe[src]", "src", ImportKind::Url),
];

const SELECTOR_CAP: usize = TAG_HANDLERS.len() + 3;

#[inline]
fn lol_err<E>(_: E) -> Error {
    crate::Error::Fail
}

/// `element_content_handlers` entry with only the element slot populated —
/// the only shape this processor registers (leaving the comment/text slots
/// empty lets lol-html skip lexing that content).
fn element_entry<'h>(
    selector: &str,
    element: lol_html::ElementHandler<'h>,
) -> Result<
    (
        Cow<'static, lol_html::Selector>,
        lol_html::ElementContentHandlers<'h>,
    ),
    Error,
> {
    Ok((
        Cow::Owned(selector.parse().map_err(lol_err)?),
        lol_html::ElementContentHandlers {
            element: Some(element),
            comments: None,
            text: None,
        },
    ))
}

impl<T: HTMLProcessorHandler, const VISIT_DOCUMENT_TAGS: bool>
    HTMLProcessor<T, VISIT_DOCUMENT_TAGS>
{
    pub(crate) fn run(this: &mut T, input: &[u8]) -> Result<(), Error> {
        // Every handler closure and the output sink capture this raw pointer
        // so one `&mut T` can service them all; `this` is not reborrowed
        // until the rewriter holding those closures is gone.
        let this_ptr: *mut T = this;

        let mut element_content_handlers = Vec::with_capacity(SELECTOR_CAP);

        for tag_info in TAG_HANDLERS {
            let on_element: lol_html::ElementHandler<'_> = Box::new(
                move |element: &mut Element<'_, '_>| -> lol_html::HandlerResult {
                    if !tag_info.url_attribute.is_empty()
                        && element.has_attribute(tag_info.url_attribute)
                    {
                        let value = element
                            .get_attribute(tag_info.url_attribute)
                            .unwrap_or_default();
                        if !value.is_empty() {
                            bun_core::scoped_log!(HTMLScanner, "{} {}", tag_info.selector, value);
                            // SAFETY: `this_ptr` was derived from `run`'s `&mut T`,
                            // which is not reborrowed while the rewriter — the only
                            // holder of these closures — is alive.
                            unsafe {
                                (*this_ptr).on_tag(
                                    element,
                                    value.as_bytes(),
                                    tag_info.url_attribute.as_bytes(),
                                    tag_info.kind,
                                );
                            }
                        }
                    }
                    Ok(())
                },
            );
            element_content_handlers.push(element_entry(tag_info.selector, on_element)?);
        }

        if VISIT_DOCUMENT_TAGS {
            for (which, tag) in ["body", "head", "html"].into_iter().enumerate() {
                let on_element: lol_html::ElementHandler<'_> = Box::new(
                    move |element: &mut Element<'_, '_>| -> lol_html::HandlerResult {
                        // SAFETY: see `on_tag` above.
                        let stop = unsafe {
                            match which {
                                0 => (*this_ptr).on_body_tag(element),
                                1 => (*this_ptr).on_head_tag(element),
                                _ => (*this_ptr).on_html_tag(element),
                            }
                        };
                        if stop {
                            // The exact text lol-html's C API attached to a
                            // LOL_HTML_STOP directive (c-api/rewriter_builder.rs).
                            Err("The rewriter has been stopped.".into())
                        } else {
                            Ok(())
                        }
                    },
                );
                element_content_handlers.push(element_entry(tag, on_element)?);
            }
        }

        let settings = lol_html::Settings {
            element_content_handlers,
            encoding: lol_html::AsciiCompatibleEncoding::utf_8(),
            memory_settings: lol_html::MemorySettings {
                preallocated_parsing_buffer_size: (input.len() / 4).max(1024),
                max_allowed_memory_usage: 1024 * 1024 * 10,
            },
            strict: false,
            ..lol_html::Settings::new()
        };

        // lol-html signals end-of-document with one zero-length chunk; the
        // C-API sink routed that to a no-op `done()`, never to `on_write_html`.
        let output_sink = OutputSink::Callback(Box::new(move |chunk: &[u8]| {
            if !chunk.is_empty() {
                // SAFETY: see `on_tag` above.
                unsafe { (*this_ptr).on_write_html(chunk) }
            }
        }));

        // The rewriter — the sole holder of `this_ptr`-derived aliases — is
        // consumed (or dropped on a failed `write`) inside this closure, so
        // reasserting the original `&mut T` borrow afterward is sound.
        let res: Result<(), lol_html::errors::RewritingError> = (|| {
            let mut rewriter = lol_html::HtmlRewriter::new(settings, output_sink);
            rewriter.write(input)?;
            rewriter.end()
        })();

        if let Err(err) = &res {
            this.on_html_parse_error(err.to_string().as_bytes());
        }
        res.map_err(lol_err)
    }
}
