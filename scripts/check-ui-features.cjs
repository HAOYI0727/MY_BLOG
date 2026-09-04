const { existsSync, readFileSync } = require("node:fs");
const { join, resolve } = require("node:path");
const yaml = require("js-yaml");

const outputDir = resolve(process.argv[2] || "dist");
const projectRoot = resolve(__dirname, "..");

function fail(message) {
    console.error(`❌ ${message}`);
    process.exitCode = 1;
}

function requireFile(path, label) {
    if (!existsSync(path)) {
        fail(`${label}不存在：${path}`);
        return false;
    }
    return true;
}

const pageContracts = [
    ["index.html", "首页"],
    [join("archive", "index.html"), "归档页"],
    [join("about", "index.html"), "内页"],
];

const featureMarkers = [
    ['href="/archive/"', "Archive 直达入口"],
    ['id="directory-left"', "左侧 Directory"],
    ['id="statistics-right"', "右侧 Statistics"],
    ["mobile-sidebar-drawer", "移动端侧栏抽屉"],
    ["music-player fixed", "音乐播放器"],
    ['id="search-switch"', "搜索"],
    ['id="display-settings-switch"', "主题颜色"],
    ['id="scheme-switch"', "亮暗模式"],
    ['id="wallpaper-mode-switch"', "背景风格"],
    ['id="translate-switch"', "语言翻译"],
    ['aria-keyshortcuts="Control+K Meta+K /"', "搜索快捷键"],
    ['id="page-overlay-container"', "页面级弹窗容器"],
];

for (const [relativePath, pageName] of pageContracts) {
    const pagePath = join(outputDir, relativePath);
    if (!requireFile(pagePath, pageName)) continue;

    const html = readFileSync(pagePath, "utf8");
    for (const [marker, featureName] of featureMarkers) {
        if (!html.includes(marker)) {
            fail(`${pageName}缺少${featureName}`);
        }
    }

    const tagModalCount = (html.match(/class="tag-group-modal"/g) || []).length;
    if (tagModalCount !== 1) {
        fail(`${pageName}应仅输出一个全局标签面板，实际为 ${tagModalCount} 个`);
    }
}

const homeHtmlPath = join(outputDir, "index.html");
if (requireFile(homeHtmlPath, "首页继续阅读入口")) {
    const homeHtml = readFileSync(homeHtmlPath, "utf8");
    if (!homeHtml.includes('id="continue-reading"')) fail("首页缺少继续阅读卡片");
    if (!homeHtml.includes('id="progress-bar-wrapper"')) fail("全局阅读进度条未输出");
    if (!homeHtml.includes('id="welcome-gateway"')) fail("首页缺少沉浸式欢迎入口");
    if (!homeHtml.includes('id="welcome-enter"')) fail("欢迎入口缺少进入网站按钮");
    if ((homeHtml.match(/class="welcome-latest group"/g) || []).length !== 3) fail("欢迎页应展示三篇最新文章");
    if ((homeHtml.match(/class="welcome-topic"/g) || []).length < 4) fail("欢迎页缺少全部分类入口");
    if ((homeHtml.match(/class="welcome-feature"/g) || []).length !== 4) fail("欢迎页快捷导航不完整");
    if (!homeHtml.includes('class="welcome-topic-bubbles"')) fail("欢迎页分类未使用气泡布局");
    if (!homeHtml.includes('class="welcome-scanline')) fail("欢迎页缺少科技扫描光效果");
    if (homeHtml.includes("data-featured")) fail("欢迎页最新文章卡片尺寸不统一");
}

const categoryHtmlPath = join(outputDir, "category", "Machine_Learning", "index.html");
if (requireFile(categoryHtmlPath, "分类文章卡片页")) {
    const categoryHtml = readFileSync(categoryHtmlPath, "utf8");
    if (!categoryHtml.includes("post-collection-grid")) fail("分类页缺少三列文章卡片网格");
}

const representativePostPath = join(outputDir, "posts", "ai_alignment", "rlvr", "index.html");
if (requireFile(representativePostPath, "文章全屏壁纸页")) {
    const postHtml = readFileSync(representativePostPath, "utf8");
    if (!postHtml.includes('id="page-wallpaper-carrier" data-wallpaper-mode="fullscreen"')) {
        fail("文章页未声明全屏壁纸模式");
    }
    if (!postHtml.includes('id="banner-wrapper" class="absolute z-10 w-full transition-all duration-600 overflow-hidden hidden"')) {
        fail("文章页首屏仍显示 Banner");
    }
    if (!postHtml.includes('class="related-posts card-base')) fail("文章页缺少相关文章推荐");
    if (!postHtml.includes("data-reading-status")) fail("文章页缺少剩余阅读时间状态");
    if (!postHtml.includes('href="/?view=home"')) fail("文章返回按钮未直达主页内容");
    if ((postHtml.match(/class="collection-card group/g) || []).length < 3) {
        fail("文章页相关文章推荐不足三篇");
    }
}

requireFile(join(outputDir, "pagefind", "pagefind.js"), "Pagefind 搜索索引");

const sourceContracts = [
    [
        join(projectRoot, "twilight.config.yaml"),
        [
            ['lang: "zh_hans"', "站点源语言为简体中文"],
            ["enable: true", "翻译功能开关"],
            ['service: "client.edge"', "客户端翻译服务"],
        ],
    ],
    [
        join(projectRoot, "src", "components", "navbar", "translator.svelte"),
        [
            ["I18nKey.languageTranslation", "翻译按钮本地化标签"],
            ["I18nKey.selectLanguage", "语言面板本地化标题"],
            ["getSupportedTranslateLanguages", "完整翻译语言列表"],
        ],
    ],
    [
        join(projectRoot, "src", "components", "WelcomeGateway.astro"),
        [
            ["syncWithRoute", "欢迎页随首页路由重复展示"],
            ["consumeDirectHomeIntent", "文章返回主页时跳过欢迎页一次"],
            ['window.swup.hooks.on("page:view", syncWithRoute)', "无刷新返回首页时重开欢迎页"],
            ["data-welcome-link", "欢迎页内容导航"],
            ["allCategories.map", "欢迎页完整分类气泡"],
            ["is-opening", "欢迎页进入主页揭幕动画"],
            ['event.key === "Escape"', "欢迎页键盘退出"],
            ['event.key === "Tab"', "欢迎页焦点循环"],
            ["prefers-reduced-motion", "欢迎页减少动态效果支持"],
        ],
    ],
    [
        join(projectRoot, "src", "utils", "url.ts"),
        [
            ['url(`/category/${encodeURIComponent(label)}/`)', "分类链接直达卡片页"],
        ],
    ],
    [
        join(projectRoot, "src", "utils", "wallpaper.ts"),
        [
            ["getPageWallpaperMode", "页面级壁纸模式"],
            ["getEffectiveWallpaperMode", "壁纸偏好与页面模式协调"],
            ["twilight:wallpaper-mode-applied", "壁纸切换状态同步"],
            ["Banner 只负责首屏展示", "Banner 模式正文延续全屏壁纸"],
        ],
    ],
    [
        join(projectRoot, "src", "components", "navbar", "search.svelte"),
        [
            ["requestSequence", "搜索异步竞态保护"],
            ['status = "unavailable"', "搜索不可用反馈"],
            ["response.results.slice(0, 12)", "搜索结果数量限制"],
            ["selectedIndex", "搜索结果键盘选中状态"],
            ['event.key === "ArrowDown"', "搜索结果方向键导航"],
            ['role={status === "ready" ? "listbox"', "搜索结果无障碍列表语义"],
        ],
    ],
    [
        join(projectRoot, "src", "components", "navbar", "navMenu.svelte"),
        [
            ["mobile-nav-open", "移动导航背景滚动锁定"],
            ["backdrop-blur-[2px]", "移动导航遮罩"],
        ],
    ],
    [
        join(projectRoot, "src", "pages", "posts", "[...slug].astro"),
        [
            ["data-reading-progress", "文章区域阅读进度"],
            ['data-share="native"', "系统原生分享"],
            ["data-post-share", "文章分享数据载体"],
            ['href={url("/?view=home")}', "文章 Back to 直返首页内容"],
            ["data-enter-home-directly", "文章返回入口跳过欢迎页一次"],
            ["I18nKey.backToHome", "文章返回主页文案"],
        ],
    ],
    [
        join(projectRoot, "src", "utils", "postShare.ts"),
        [
            ["initPostShare", "文章分享重复初始化"],
            ["__twilightPostShareCleanup", "文章分享监听清理"],
            ["__twilightPostPageCleanup", "文章时间监听清理"],
            ["QQ 分享窗口被浏览器拦截", "QQ 分享失败反馈"],
            ["二维码加载失败", "微信二维码失败反馈"],
            ["navigator.clipboard", "剪贴板复制"],
        ],
    ],
    [
        join(projectRoot, "src", "utils", "readingContinuity.ts"),
        [
            ["twilight:reading-progress:v1", "跨会话阅读进度存储"],
            ["resumeUrl.searchParams.set", "继续阅读定位入口"],
            ["window.__twilightReadingContinuityCleanup", "阅读监听器清理"],
            ["data-reading-status", "剩余阅读时间更新"],
        ],
    ],
    [
        join(projectRoot, "src", "utils", "markdown.ts"),
        [
            ["decorateSectionLinks", "文章章节链接增强"],
            ["data-heading-anchor", "章节链接复制按钮"],
            ["markdownActionsBound", "Markdown 事件防重复绑定"],
        ],
    ],
    [
        join(projectRoot, "src", "components", "sidebar", "toc.ts"),
        [
            ['e.key !== "Escape"', "浮动目录 Escape 关闭"],
            ['aria-expanded', "浮动目录展开状态"],
        ],
    ],
    [
        join(projectRoot, "src", "components", "sidebar.astro"),
        [
            ["sidebar-shell", "侧栏外壳视口粘性定位"],
            ["sidebar-scroll-region", "桌面与平板侧栏独立滚动"],
            ["topComponents.map", "侧栏顶部组件进入整体滚动流"],
            ["stickyComponents.map", "侧栏长内容进入整体滚动流"],
            ["overflow-y: auto", "侧栏纵向滚动能力"],
            ["isolation: isolate", "侧栏卡片层叠隔离"],
            ["setupSidebarWheelRouting", "侧栏滚轮路由"],
            ['[id^="statistics-"]', "Statistics 防覆盖定位约束"],
        ],
    ],
    [
        join(projectRoot, "src", "components", "sidebar", "tags.astro"),
        [
            ['data-modal-id={tagModalId}', "左右侧栏共享标签面板"],
            ['aria-expanded="false"', "标签面板触发器状态"],
        ],
    ],
    [
        join(projectRoot, "src", "components", "sidebar", "TagGroupModal.astro"),
        [
            ["ensureTagsRendered", "完整标签按需渲染"],
            ["filterTags", "完整标签搜索筛选"],
            ["tag-group-data", "标签数据安全传递"],
        ],
    ],
    [
        join(projectRoot, "src", "styles", "transition.css"),
        [
            ["内容默认必须可见", "首屏内容可见性安全策略"],
        ],
    ],
    [
        join(projectRoot, "astro.config.mjs"),
        [
            ["persistAssets: true", "Swup 跨页面样式保留"],
            ["awaitAssets: true", "Swup 新页面样式等待"],
        ],
    ],
];

for (const [sourcePath, markers] of sourceContracts) {
    if (!requireFile(sourcePath, "体验功能源码")) continue;
    const source = readFileSync(sourcePath, "utf8");
    for (const [marker, featureName] of markers) {
        if (!source.includes(marker)) fail(`源码缺少${featureName}`);
    }
}

const linkPresetSource = readFileSync(
    join(projectRoot, "src", "constants", "link-presets.ts"),
    "utf8",
);
const archivePresetStart = linkPresetSource.indexOf("[LinkPreset.Archive]");
const projectsPresetStart = linkPresetSource.indexOf("[LinkPreset.Projects]", archivePresetStart);
const archivePresetBlock = linkPresetSource.slice(archivePresetStart, projectsPresetStart);
if (archivePresetStart < 0 || projectsPresetStart < 0 || archivePresetBlock.includes("children")) {
    fail("Archive 必须是无下拉菜单的 All Posts 直达入口");
}

const configPath = join(projectRoot, "twilight.config.yaml");
const config = yaml.load(readFileSync(configPath, "utf8"));
const musicPlayer = config?.musicPlayer;

if (!musicPlayer?.enable) {
    fail("musicPlayer.enable 必须保持为 true");
} else {
    const playlist = musicPlayer.local?.playlist || [];
    if (playlist.length === 0) {
        fail("本地音乐播放列表为空");
    }

    for (const track of playlist) {
        for (const field of ["url", "cover", "lrc"]) {
            const configuredPath = track[field];
            if (!configuredPath) continue;
            const assetPath = join(outputDir, configuredPath.replace(/^\//, ""));
            requireFile(assetPath, `音乐 #${track.id} 的 ${field} 资源`);
        }
    }
}

if (process.exitCode) {
    process.exit(process.exitCode);
}

console.log("✅ UI 功能契约检查通过：导航、侧栏、音乐、搜索和外观设置均已输出");
