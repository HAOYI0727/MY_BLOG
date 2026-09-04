let markdownActionsBound = false;

function fallbackCopy(text: string): boolean {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.setAttribute("readonly", "");
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "0";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
        return document.execCommand("copy");
    } catch {
        return false;
    } finally {
        textArea.remove();
    }
}

async function copyText(text: string): Promise<boolean> {
    try {
        if (window.isSecureContext && navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch {
        // Fall through to the compatibility path below.
    }
    return fallbackCopy(text);
}

function sectionCopyLabels() {
    const lang = document.documentElement.lang.toLowerCase();
    if (lang.startsWith("zh")) return { copy: "复制本节链接", copied: "本节链接已复制" };
    if (lang.startsWith("ja")) return { copy: "セクションリンクをコピー", copied: "リンクをコピーしました" };
    return { copy: "Copy section link", copied: "Section link copied" };
}

function decorateSectionLinks() {
    const labels = sectionCopyLabels();
    document.querySelectorAll<HTMLElement>(".markdown-content :is(h2, h3, h4, h5, h6)[id]").forEach((heading) => {
        if (heading.querySelector("[data-heading-anchor]")) return;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "heading-anchor-btn";
        button.dataset.headingAnchor = heading.id;
        button.title = labels.copy;
        button.setAttribute("aria-label", `${labels.copy}：${heading.textContent?.trim() || heading.id}`);
        button.innerHTML = '<span aria-hidden="true">#</span>';
        heading.appendChild(button);
    });
}

/**
 * Markdown 相关交互逻辑
 * 包括代码块复制和折叠功能
 * 使用事件委托，确保在 Swup 无刷新跳转后依然有效
 */

export function initMarkdownActions() {
    if (typeof document === "undefined") return;
    decorateSectionLinks();
    if (markdownActionsBound) return;
    markdownActionsBound = true;

    document.addEventListener("click", function (e: MouseEvent) {
        const target = e.target as Element | null;
        if (!target) return;

        const headingAnchor = target.closest<HTMLButtonElement>("[data-heading-anchor]");
        if (headingAnchor) {
            e.preventDefault();
            const id = headingAnchor.dataset.headingAnchor;
            if (!id) return;
            const sectionUrl = new URL(window.location.href);
            sectionUrl.hash = id;
            const labels = sectionCopyLabels();
            const defaultAriaLabel = headingAnchor.getAttribute("aria-label") || labels.copy;
            copyText(sectionUrl.toString()).then((copied) => {
                if (!copied) return;
                const oldTimer = Number(headingAnchor.dataset.timeoutId || 0);
                if (oldTimer) window.clearTimeout(oldTimer);
                headingAnchor.classList.add("success");
                headingAnchor.title = labels.copied;
                headingAnchor.setAttribute("aria-label", labels.copied);
                headingAnchor.innerHTML = '<span aria-hidden="true">✓</span>';
                const timer = window.setTimeout(() => {
                    headingAnchor.classList.remove("success");
                    headingAnchor.title = labels.copy;
                    headingAnchor.setAttribute("aria-label", defaultAriaLabel);
                    headingAnchor.innerHTML = '<span aria-hidden="true">#</span>';
                }, 1600);
                headingAnchor.dataset.timeoutId = String(timer);
            });
            return;
        }

        // 1. 处理复制按钮点击
        if (target.classList.contains("copy-btn") || target.closest(".copy-btn")) {
            const btn = target.classList.contains("copy-btn") ? target : target.closest(".copy-btn");
            if (!btn) return;
            
            const codeEle = btn.parentElement?.querySelector("code");

            // 精确的代码提取逻辑
            let code = '';
            if (codeEle) {
                // 获取所有代码行元素
                const lineElements = codeEle.querySelectorAll('span.line');
                // 对于有行结构的代码块，精确处理每一行
                if (lineElements.length > 0) {
                    const lines: string[] = [];
                    for (let i = 0; i < lineElements.length; i++) {
                        const lineElement = lineElements[i];
                        const lineText = lineElement.textContent || '';
                        lines.push(lineText);
                    }
                    code = lines.join('\n');
                } else {
                    const codeElements = codeEle.querySelectorAll('.code:not(summary *)');
                    if (codeElements.length > 0) {
                        const lines: string[] = [];
                        for (let i = 0; i < codeElements.length; i++) {
                            const el = codeElements[i];
                            const lineText = el.textContent || '';
                            lines.push(lineText);
                        }
                        code = lines.join('\n');
                    } else {
                        code = codeEle.textContent || '';
                    }
                }
            }

            // 处理连续空行
            code = code.replace(/\n\n\n+/g, function(match) {
                const newlineCount = match.length;
                const emptyLineCount = newlineCount - 1;
                let resultEmptyLines: number;
                if (emptyLineCount % 2 === 0) {
                    resultEmptyLines = emptyLineCount / 2;
                } else {
                    resultEmptyLines = Math.floor((emptyLineCount + 1) / 2);
                }
                if (resultEmptyLines < 1) resultEmptyLines = 1;
                return '\n'.repeat(resultEmptyLines + 1);
            });

            // 调用复制函数
            copyText(code).then((copied) => {
                if (!copied) throw new Error("复制失败");
                const timeoutId = btn.getAttribute("data-timeout-id");
                if (timeoutId) {
                    clearTimeout(parseInt(timeoutId));
                }
                btn.classList.add("success");
                const newTimeoutId = setTimeout(() => {
                    btn.classList.remove("success");
                }, 1000);
                btn.setAttribute("data-timeout-id", newTimeoutId.toString());
            }).catch(err => {
                console.error('复制失败:', err);
            });
        }

        // 2. 处理折叠按钮点击
        if (target.classList.contains("collapse-btn") || target.closest(".collapse-btn")) {
            const btn = target.classList.contains("collapse-btn") ? target : target.closest(".collapse-btn");
            const codeBlock = btn?.closest(".expressive-code");
            if (codeBlock) {
                codeBlock.classList.toggle("collapsed");
                codeBlock.classList.toggle("expanded");
            }
        }
    });
}
