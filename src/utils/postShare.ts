function fallbackCopy(text: string): boolean {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    textarea.style.top = "0";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    let copied = false;
    try {
        copied = document.execCommand("copy");
    } catch {
        copied = false;
    } finally {
        textarea.remove();
    }
    return copied;
}

export function initPostShare(): void {
    window.__twilightPostShareCleanup?.();

    const root = document.querySelector<HTMLElement>("[data-post-share]");
    if (!root) return;

    const controller = new AbortController();
    const { signal } = controller;
    const timers = new Set<number>();
    const modal = document.getElementById("wechat-modal");
    const qrImage = document.getElementById("wechat-qr-image") as HTMLImageElement | null;
    const qrStatus = document.querySelector<HTMLElement>("[data-wechat-qr-status]");
    const wechatButton = root.querySelector<HTMLElement>("[data-share='wechat']");
    let lastModalFocus: HTMLElement | null = null;

    const shareUrl = root.dataset.shareUrl || window.location.href;
    const shareTitle = root.dataset.shareTitle || document.title;
    const shareDescription = root.dataset.shareDescription || shareTitle;

    let modifiedTimer: number | undefined;
    const updateLastModified = () => {
        const source = document.getElementById("last-modified");
        const output = document.getElementById("modifiedtime");
        if (!source || !output || !source.dataset.lastModified || !source.dataset.translations) return;

        let translations: Record<string, Record<string, string>>;
        try {
            translations = JSON.parse(source.dataset.translations);
        } catch {
            return;
        }

        const diffSeconds = Math.max(0, Math.floor((Date.now() - new Date(source.dataset.lastModified).getTime()) / 1000));
        const days = Math.floor(diffSeconds / 86400);
        const years = Math.floor(days / 365);
        const months = Math.floor((days % 365) / 30);
        const remainingDays = days % 30;
        const hours = Math.floor((diffSeconds % 86400) / 3600);
        const minutes = Math.floor((diffSeconds % 3600) / 60);
        const seconds = diffSeconds % 60;
        const defaultLang = source.dataset.defaultLang || "";
        let currentLang = defaultLang;
        try {
            currentLang = localStorage.getItem("selected-language") || defaultLang;
        } catch {
            // Storage may be unavailable in private browsing contexts.
        }
        const words = translations[currentLang] || translations[defaultLang] || Object.values(translations)[0];
        if (!words) return;

        const parts = [words.lastModifiedPrefix];
        if (years) parts.push(`${years} ${words.year}`);
        if (months) parts.push(`${months} ${words.month}`);
        if (remainingDays) parts.push(`${remainingDays} ${words.day}`);
        parts.push(`${hours} ${words.hour}`, `${String(minutes).padStart(2, "0")} ${words.minute}`, `${String(seconds).padStart(2, "0")} ${words.second}`);
        output.textContent = parts.join(" ");
    };

    updateLastModified();
    if (document.getElementById("last-modified")) {
        modifiedTimer = window.setInterval(updateLastModified, 1000);
    }

    const schedule = (callback: () => void, delay: number) => {
        const timer = window.setTimeout(() => {
            timers.delete(timer);
            callback();
        }, delay);
        timers.add(timer);
        return timer;
    };

    const showToast = (message: string, tone: "success" | "warning" = "success") => {
        document.getElementById("share-toast")?.remove();
        const toast = document.createElement("div");
        toast.id = "share-toast";
        toast.setAttribute("role", "status");
        toast.setAttribute("aria-live", "polite");
        toast.dataset.tone = tone;
        toast.textContent = message;
        document.body.appendChild(toast);
        requestAnimationFrame(() => toast.classList.add("is-visible"));
        schedule(() => {
            toast.classList.remove("is-visible");
            schedule(() => toast.remove(), 260);
        }, 2400);
    };

    const showCopiedState = (button?: HTMLElement | null) => {
        const label = button?.querySelector<HTMLElement>("[data-share-label]");
        if (!label) return;
        const original = label.dataset.defaultLabel || label.textContent || "复制链接";
        label.textContent = "已复制";
        button?.classList.add("is-success");
        schedule(() => {
            label.textContent = original;
            button?.classList.remove("is-success");
        }, 1800);
    };

    const copyLink = async (button?: HTMLElement | null): Promise<boolean> => {
        let copied = false;
        try {
            if (window.isSecureContext && navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(shareUrl);
                copied = true;
            } else {
                copied = fallbackCopy(shareUrl);
            }
        } catch {
            copied = fallbackCopy(shareUrl);
        }

        if (copied) {
            showCopiedState(button);
            showToast("链接已复制到剪贴板");
        } else {
            showToast("复制失败，请从地址栏手动复制链接", "warning");
        }
        return copied;
    };

    const closeWechatModal = (restoreFocus = true) => {
        if (!modal || modal.hidden) return;
        modal.classList.remove("is-open");
        wechatButton?.setAttribute("aria-expanded", "false");
        schedule(() => {
            modal.hidden = true;
            if (restoreFocus) lastModalFocus?.focus({ preventScroll: true });
        }, 260);
    };

    const shareToWechat = () => {
        if (!modal || !qrImage) {
            showToast("微信分享面板暂时不可用", "warning");
            return;
        }

        lastModalFocus = document.activeElement as HTMLElement | null;
        modal.hidden = false;
        wechatButton?.setAttribute("aria-expanded", "true");
        qrImage.hidden = true;
        qrStatus?.removeAttribute("hidden");
        if (qrStatus) qrStatus.textContent = "二维码生成中…";

        qrImage.onload = () => {
            qrImage.hidden = false;
            qrStatus?.setAttribute("hidden", "");
        };
        qrImage.onerror = () => {
            qrImage.hidden = true;
            if (qrStatus) {
                qrStatus.removeAttribute("hidden");
                qrStatus.textContent = "二维码加载失败，请使用下方复制链接";
            }
        };
        qrImage.src = `https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=${encodeURIComponent(shareUrl)}`;

        requestAnimationFrame(() => {
            modal.classList.add("is-open");
            document.getElementById("wechat-modal-close")?.focus({ preventScroll: true });
        });
    };

    const shareToQQ = () => {
        const qqUrl = `https://connect.qq.com/widget/shareqq/index.html?url=${encodeURIComponent(shareUrl)}&title=${encodeURIComponent(shareTitle)}&desc=${encodeURIComponent(shareDescription)}&summary=${encodeURIComponent(shareDescription)}`;
        const popup = window.open(qqUrl, "qq-share", "popup=yes,width=680,height=600");
        if (popup) {
            popup.opener = null;
            popup.focus();
            showToast("已打开 QQ 分享小窗口");
        } else {
            showToast("QQ 分享窗口被浏览器拦截，请允许弹窗后重试", "warning");
        }
    };

    const shareNative = async (button: HTMLElement) => {
        if (!navigator.share) {
            await copyLink(button);
            return;
        }
        try {
            await navigator.share({ title: shareTitle, text: shareDescription, url: shareUrl });
        } catch (error) {
            if (error instanceof DOMException && error.name === "AbortError") return;
            await copyLink(button);
        }
    };

    root.querySelectorAll<HTMLElement>("[data-share]").forEach((button) => {
        button.addEventListener("click", async (event) => {
            event.preventDefault();
            switch (button.dataset.share) {
                case "qq":
                    shareToQQ();
                    break;
                case "wechat":
                    shareToWechat();
                    break;
                case "copy":
                    await copyLink(button);
                    break;
                case "native":
                    await shareNative(button);
                    break;
            }
        }, { signal });
    });

    const nativeShareButton = root.querySelector<HTMLElement>("[data-share='native']");
    if (nativeShareButton && "share" in navigator) {
        nativeShareButton.classList.remove("hidden");
        nativeShareButton.classList.add("flex");
    }

    document.getElementById("wechat-modal-close")?.addEventListener("click", () => closeWechatModal(), { signal });
    modal?.addEventListener("click", (event) => {
        if (event.target === modal) closeWechatModal();
    }, { signal });
    document.addEventListener("pointerdown", (event) => {
        if (!modal || modal.hidden || !(event.target instanceof Element)) return;
        if (!modal.contains(event.target) && !event.target.closest("[data-share='wechat']")) {
            closeWechatModal(false);
        }
    }, { signal });
    document.getElementById("wechat-copy-link-btn")?.addEventListener("click", async (event) => {
        const copied = await copyLink(event.currentTarget as HTMLElement);
        if (copied) schedule(() => closeWechatModal(), 420);
    }, { signal });

    document.addEventListener("keydown", (event) => {
        if (!modal || modal.hidden) return;
        if (event.key === "Escape") {
            event.preventDefault();
            closeWechatModal();
            return;
        }
    }, { signal });

    const cleanup = () => {
        controller.abort();
        timers.forEach((timer) => window.clearTimeout(timer));
        timers.clear();
        if (qrImage) {
            qrImage.onload = null;
            qrImage.onerror = null;
            qrImage.removeAttribute("src");
        }
        document.getElementById("share-toast")?.remove();
        if (modal) {
            modal.hidden = true;
            modal.classList.remove("is-open");
        }
        wechatButton?.setAttribute("aria-expanded", "false");
        if (window.__twilightPostShareCleanup === cleanup) {
            window.__twilightPostShareCleanup = undefined;
        }
    };
    window.__twilightPostShareCleanup = cleanup;

    const pageCleanup = () => {
        if (modifiedTimer) window.clearInterval(modifiedTimer);
        if (window.__twilightPostPageCleanup === pageCleanup) {
            window.__twilightPostPageCleanup = undefined;
        }
    };
    window.__twilightPostPageCleanup = pageCleanup;
}
