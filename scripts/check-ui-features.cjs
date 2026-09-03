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
    ['aria-keyshortcuts="Control+K Meta+K /"', "搜索快捷键"],
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

requireFile(join(outputDir, "pagefind", "pagefind.js"), "Pagefind 搜索索引");

const sourceContracts = [
    [
        join(projectRoot, "src", "components", "navbar", "search.svelte"),
        [
            ["requestSequence", "搜索异步竞态保护"],
            ['status = "unavailable"', "搜索不可用反馈"],
            ["response.results.slice(0, 12)", "搜索结果数量限制"],
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
            ["__twilightPostPageCleanup", "文章交互监听清理"],
            ["postContainerElement.offsetHeight", "文章区域阅读进度"],
            ['href={url("/")}', "文章 Back to 直返首页"],
            ["I18nKey.backTo", "文章返回入口文案"],
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
