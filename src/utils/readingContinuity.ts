const STORAGE_KEY = "twilight:reading-progress:v1";
const MIN_RESUMABLE_PROGRESS = 0.02;
const COMPLETED_PROGRESS = 0.96;
const MAX_AGE_MS = 90 * 24 * 60 * 60 * 1000;

type ReadingRecord = {
    url: string;
    title: string;
    progress: number;
    updatedAt: number;
};

function readRecord(): ReadingRecord | null {
    try {
        const value = localStorage.getItem(STORAGE_KEY);
        if (!value) return null;
        const record = JSON.parse(value) as Partial<ReadingRecord>;
        if (
            typeof record.url !== "string" ||
            typeof record.title !== "string" ||
            typeof record.progress !== "number" ||
            typeof record.updatedAt !== "number" ||
            record.progress < MIN_RESUMABLE_PROGRESS ||
            record.progress >= COMPLETED_PROGRESS ||
            Date.now() - record.updatedAt > MAX_AGE_MS
        ) {
            localStorage.removeItem(STORAGE_KEY);
            return null;
        }
        return record as ReadingRecord;
    } catch {
        return null;
    }
}

function writeRecord(record: ReadingRecord) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(record));
    } catch {
        // Reading continuity is optional when storage is unavailable.
    }
}

function clearRecord(url?: string) {
    try {
        const current = readRecord();
        if (!url || !current || current.url === url) localStorage.removeItem(STORAGE_KEY);
    } catch {
        // Ignore unavailable storage.
    }
}

function renderContinueReading(signal: AbortSignal) {
    const card = document.getElementById("continue-reading") as HTMLElement | null;
    if (!card) return;

    const record = readRecord();
    if (!record || record.url === window.location.pathname) {
        card.hidden = true;
        return;
    }

    const title = card.querySelector<HTMLElement>("[data-continue-title]");
    const progressText = card.querySelector<HTMLElement>("[data-continue-progress-text]");
    const progress = card.querySelector<HTMLElement>("[data-continue-progress]");
    const fill = card.querySelector<HTMLElement>("[data-continue-progress-fill]");
    const links = card.querySelectorAll<HTMLAnchorElement>("[data-continue-link], [data-continue-action]");
    const dismiss = card.querySelector<HTMLButtonElement>("[data-continue-dismiss]");
    const percentage = Math.max(1, Math.min(95, Math.round(record.progress * 100)));

    if (title) title.textContent = record.title;
    if (progressText) progressText.textContent = `${percentage}%`;
    if (progress) progress.setAttribute("aria-valuenow", String(percentage));
    if (fill) fill.style.transform = `scaleX(${record.progress})`;
    const resumeUrl = new URL(record.url, window.location.origin);
    resumeUrl.searchParams.set("resume", "1");
    links.forEach((link) => (link.href = `${resumeUrl.pathname}${resumeUrl.search}${resumeUrl.hash}`));
    card.hidden = false;

    dismiss?.addEventListener("click", () => {
        clearRecord(record.url);
        card.hidden = true;
    }, { signal });
}

export function initReadingContinuity() {
    if (typeof window === "undefined") return;

    window.__twilightReadingContinuityCleanup?.();

    const controller = new AbortController();
    const { signal } = controller;
    const article = document.querySelector<HTMLElement>("[data-reading-progress]");
    const progressWrapper = document.getElementById("progress-bar-wrapper");
    const progressFill = document.getElementById("progress-bar-fill");
    const readingStatus = article?.querySelector<HTMLElement>("[data-reading-status]");
    let lastProgress = 0;
    let lastStoredAt = 0;
    let ticking = false;

    const resetProgressBar = () => {
        if (progressFill) progressFill.style.transform = "scaleX(0)";
        progressWrapper?.classList.add("opacity-0");
        progressWrapper?.classList.remove("opacity-100");
        progressWrapper?.setAttribute("aria-valuenow", "0");
        progressWrapper?.setAttribute("aria-hidden", "true");
    };

    const persist = (force = false) => {
        if (!article) return;
        const url = article.dataset.readingUrl;
        const title = article.dataset.readingTitle;
        if (!url || !title) return;

        if (lastProgress >= COMPLETED_PROGRESS) {
            clearRecord(url);
            return;
        }
        if (lastProgress < MIN_RESUMABLE_PROGRESS) return;

        const now = Date.now();
        if (!force && now - lastStoredAt < 900) return;
        lastStoredAt = now;
        writeRecord({ url, title, progress: lastProgress, updatedAt: now });
    };

    const update = (force = false) => {
        if (!article || !progressWrapper || !progressFill) {
            resetProgressBar();
            return;
        }

        const articleTop = article.getBoundingClientRect().top + window.scrollY;
        const readableDistance = Math.max(article.offsetHeight - window.innerHeight, 1);
        const rawProgress = (window.scrollY - articleTop) / readableDistance;
        lastProgress = Math.min(Math.max(rawProgress, 0), 1);
        const percentage = Math.round(lastProgress * 100);
        progressFill.style.transform = `scaleX(${lastProgress})`;
        progressWrapper.classList.toggle("opacity-0", rawProgress <= 0);
        progressWrapper.classList.toggle("opacity-100", rawProgress > 0);
        progressWrapper.setAttribute("aria-valuenow", String(percentage));
        progressWrapper.setAttribute("aria-hidden", "false");
        if (readingStatus) {
            const totalMinutes = Math.max(1, Math.ceil(Number(article.dataset.readingTotalMinutes) || 1));
            const remainingMinutes = Math.max(0, Math.ceil(totalMinutes * (1 - lastProgress)));
            const progressLabel = readingStatus.dataset.progressLabel || "Reading progress";
            const remainingLabel = readingStatus.dataset.remainingLabel || "min left";
            readingStatus.textContent = `${percentage}% · ${remainingMinutes} ${remainingLabel}`;
            readingStatus.setAttribute("aria-label", `${progressLabel} ${percentage}%, ${remainingMinutes} ${remainingLabel}`);
        }
        persist(force);
    };

    const requestUpdate = () => {
        if (ticking) return;
        ticking = true;
        window.requestAnimationFrame(() => {
            update();
            ticking = false;
        });
    };

    renderContinueReading(signal);

    if (article) {
        window.addEventListener("scroll", requestUpdate, { passive: true, signal });
        window.addEventListener("resize", requestUpdate, { passive: true, signal });
        window.addEventListener("pagehide", () => update(true), { signal });
        document.addEventListener("visibilitychange", () => {
            if (document.visibilityState === "hidden") update(true);
        }, { signal });
        const stored = readRecord();
        const shouldResume = new URLSearchParams(window.location.search).get("resume") === "1" && stored?.url === window.location.pathname;
        if (shouldResume && stored) {
            // Wait for layout and persisted assets before translating the saved ratio back to a pixel position.
            window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
                const articleTop = article.getBoundingClientRect().top + window.scrollY;
                const readableDistance = Math.max(article.offsetHeight - window.innerHeight, 1);
                window.scrollTo({ top: articleTop + readableDistance * stored.progress, behavior: "auto" });
                const cleanUrl = `${window.location.pathname}${window.location.hash}`;
                window.history.replaceState(window.history.state, "", cleanUrl);
                update();
            }));
        } else {
            update();
        }
    } else {
        resetProgressBar();
    }

    window.__twilightReadingContinuityCleanup = () => {
        if (article) update(true);
        controller.abort();
        resetProgressBar();
        window.__twilightReadingContinuityCleanup = undefined;
    };
}
