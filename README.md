# ZHY's Blog

这是 Zhang Haoyi 的个人知识博客，在线地址为 [twilight.spr-aachen.com](https://twilight.spr-aachen.com/)。项目基于 Astro 5、Svelte 5、Tailwind CSS 4 和 Twilight 主题构建，用于整理 AI Agent、LLM、深度学习、机器学习与课程学习笔记，也承载项目、技能、日记、相册和个人时间线等内容。

## 功能概览

- Markdown / MDX 内容集合，支持草稿、置顶、分类、标签、封面、阅读时间和上一篇/下一篇。
- KaTeX 数学公式、Mermaid 图表、Callout、自动标题锚点和 Expressive Code 代码高亮。
- Pagefind 静态全文搜索，生产构建后自动生成索引。
- RSS、Atom、Sitemap、robots.txt、Open Graph 元信息及可选的文章 OG 图片。
- 响应式双侧栏、文章目录、主题色、明暗模式、壁纸模式与 Swup 页面过渡。
- 项目、技能、时间线、日记、相册、友链和 Bangumi 页面。
- 本地 / Meting 音乐播放器、Waline / Twikoo 评论、Umami 统计与可选看板娘。
- GitHub Pages、Vercel、Cloudflare Pages、Netlify、EdgeOne 和 Docker 部署配置。
- 可选 Decap CMS 管理后台（当前 OAuth 集成默认关闭）。

## 技术栈

| 层级 | 实现 |
| --- | --- |
| 框架 | Astro 5、TypeScript |
| 交互组件 | Svelte 5 |
| 样式 | Tailwind CSS 4、PostCSS、Stylus |
| 内容 | Astro Content Collections、Markdown / MDX、JSON |
| 内容增强 | KaTeX、Mermaid、Expressive Code、rehype / remark 插件 |
| 搜索 | Pagefind |
| 图片 | Astro Assets、Sharp |
| 部署 | GitHub Pages、Vercel、Cloudflare、Netlify、EdgeOne、Docker + Nginx |

## 快速开始

### 环境要求

- Node.js LTS（推荐 20 或更高的 LTS 版本）
- pnpm 9.14.4；版本以 `package.json#packageManager` 为准

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

开发服务器启动后访问终端显示的本地地址。开发模式不会加载 Pagefind 索引；需要验证全文搜索时请先执行生产构建，再预览构建产物。

```bash
pnpm build
pnpm preview
```

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `pnpm dev` | 生成本地图标数据并启动 Astro 开发服务器 |
| `pnpm build` | 生成图标、构建站点并创建 Pagefind 搜索索引 |
| `pnpm preview` | 本地预览生产构建 |
| `pnpm check` | 执行 Astro 类型/模板检查和 Stylus 编译检查 |
| `pnpm type-check` | 执行 Astro 类型与模板诊断 |
| `pnpm check-stylus` | 单独检查 `.styl` 和组件内联 Stylus |
| `pnpm new-post -- path/title` | 在 `src/content/posts` 下创建文章模板 |

## 项目结构

```text
.
├── public/                     # 原样复制的图片、音乐、favicon、看板娘资源
├── scripts/                    # 图标生成、文章创建、构建与样式检查脚本
├── src/
│   ├── components/             # Astro / Svelte UI 组件
│   ├── constants/              # 路由、图标与布局常量
│   ├── content/                # 文章与各展示页的数据源
│   │   ├── posts/              # Markdown / MDX 文章
│   │   ├── albums/             # 相册 JSON 与图片
│   │   ├── diary/              # 日记 JSON 与图片
│   │   ├── friends/            # 友链 JSON
│   │   ├── projects/           # 项目 JSON
│   │   ├── skills/             # 技能 JSON
│   │   └── timeline/           # 时间线 JSON
│   ├── i18n/                   # 本地界面翻译
│   ├── layouts/                # 基础布局与网格布局
│   ├── pages/                  # Astro 文件路由和 Feed / OG 接口
│   ├── plugins/                # Markdown、代码块、翻译等扩展
│   ├── styles/                 # 全局 CSS 与 Stylus
│   ├── types/                  # 配置类型
│   └── utils/                  # 内容读取、排序、URL、主题等工具
├── astro.config.mjs            # Astro、Markdown 插件及部署适配器
├── src/content.config.ts       # 文章 Content Collection Schema
├── twilight.config.yaml        # 站点的主要业务与视觉配置
├── pagefind.yml                # 搜索索引配置
├── .decap.yml                  # Decap CMS 内容模型
└── vercel.json                 # Vercel 构建、缓存与安全响应头
```

## 主要路由

| 路由 | 内容 |
| --- | --- |
| `/`、`/[page]/` | 分页文章首页 |
| `/posts/`、`/posts/[slug]/` | 文章列表与详情 |
| `/posts/[year]/[month]/` | 月度归档 |
| `/archive/` | 总归档 |
| `/category/[slug]/` | 分类筛选 |
| `/projects/`、`/skills/`、`/timeline/` | 个人展示页 |
| `/diary/`、`/diary/[slug]/` | 日记列表与详情 |
| `/albums/`、`/albums/[id]/` | 相册列表与详情 |
| `/friends/`、`/about/`、`/anime/` | 友链、关于与 Bangumi 页面 |
| `/rss.xml`、`/atom.xml` | 订阅源 |
| `/admin/` | Decap CMS 管理入口 |

## 站点配置

大多数定制项集中在 `twilight.config.yaml`：

- `site`：域名、标题、语言、字体、主题、壁纸、favicon、翻译与 OG 图片。
- `navbar`：顶级导航、下拉菜单及自定义链接。
- `sidebar`：左右侧栏组件、位置、可见路由与折叠阈值。
- `profile`、`footer`、`announcement`：个人资料、页脚和公告。
- `post`：文章卡片、代码主题、许可证和评论服务。
- `analytics`：Umami 统计。
- `particle`、`musicPlayer`、`pio`：粒子、音乐播放器和看板娘。

修改域名时务必同步更新 `site.siteURL`；RSS、Atom、Sitemap、robots.txt、canonical/OG URL 都依赖它。Decap CMS 启用时，还需要同步检查 `.decap.yml` 中的仓库、域名和 OAuth 地址。

## 写文章

可以使用脚本创建文章：

```bash
pnpm new-post -- AI_Agent/My_New_Post
```

文章位于 `src/content/posts`，路径会参与最终 slug。基础 Frontmatter 示例：

```yaml
---
title: 文章标题
published: 2026-09-03
updated: 2026-09-03
description: 用于文章卡片、搜索和 SEO 的摘要
cover: /assets/images/posts/example.png
coverInContent: false
category: AI_Agent
tags: [Agent, LLM]
lang: zh-CN
pinned: false
draft: false
comment: true
---
```

完整字段由 `src/content.config.ts` 定义：

- 基础信息：`title`、`description`、`published`、`updated`、`author`、`lang`。
- 组织信息：`category`、`tags`、`directoryTitle`、`pinned`、`routeName`。
- 展示信息：`cover`、`coverInContent`、`draft`、`comment`。
- 来源与许可：`sourceLink`、`licenseName`、`licenseUrl`。
- 页面保护：`encrypted`、`password`。这是前端保护，不应当用于发布真正敏感的内容。

生产环境会过滤 `draft: true` 的文章。RSS/Atom 也会排除加密文章；Atom 额外明确过滤草稿。

### 图片路径

- 公共资源放在 `public/`，正文中使用以 `/` 开头的路径，例如 `/assets/images/posts/example.png`。
- 与文章同目录的图片可使用 `./example.png`，构建时会交给 Astro Assets 处理。
- 文件名大小写在 Linux 部署环境中严格区分。
- 相册图片通常与相册 JSON 同目录，并使用 `./photo.JPG`。

### Markdown 扩展

项目支持数学公式、Mermaid、GitHub / Music 自定义卡片、提示块、标题锚点、代码行号、代码折叠和复制按钮。代码围栏的语言名应使用 Shiki 支持的小写标识，例如 `python`、`bash`、`text`。

## 其他内容数据

展示页使用 JSON 自动聚合，新增文件即可参与构建：

- `projects/*.json`：标题、描述、技术栈、状态、源码/演示地址和日期。
- `skills/*.json`：名称、图标、分类、等级、经验和颜色。
- `timeline/*.json`：事件类型、起止日期、技能、成果和链接。
- `diary/**/*.json`：标题、正文、日期和图片列表。
- `albums/**/*.json`：标题、封面、日期、布局、列数和照片列表。文件名默认作为相册 ID；如文件名重复，应设置唯一的 `id`。
- `friends/*.json`：名称、头像、简介、站点地址和标签。

## 搜索、评论与统计

### Pagefind

`pnpm build` 先执行 Astro 构建，再对最终目录运行 Pagefind。默认构建产物是 `dist/`；Vercel 环境使用 `.vercel/output/static/`。搜索只索引带 `data-pagefind-body` 的页面，因此索引页数少于生成页数属于正常现象。

### 评论

当前启用 Waline，配置位于 `twilight.config.yaml > post.comment`。切换到 Twikoo 时设置 `provider: twikoo` 并填写相应服务配置。评论服务端必须允许正式站点域名。

### Umami

在 `analytics.enabled` 为 `true` 时启用。密钥与 Tracking Code 可以写入部署平台环境变量：

```dotenv
UMAMI_API_KEY=
UMAMI_TRACKING_CODE=
```

## Decap CMS

管理入口会生成在 `/admin/`，但 `astro.config.mjs` 中的 OAuth 集成当前为 `enable: false`。启用前需要：

1. 在 GitHub 创建 OAuth App。
2. 复制 `.env.example` 为 `.env` 并填写凭据。
3. 校验 `.decap.yml` 中的 `repo`、`branch`、`site_domain` 和 `base_url`。
4. 将 `decapCmsOauth()` 的 `enable` 改为 `true`。

不要提交 `.env`。如果真实 OAuth Secret 曾进入 Git 历史，应立即在 GitHub 中撤销并重新生成；仅从最新提交删除文件并不能使旧 Secret 失效。

## 部署

### GitHub Pages

`.github/workflows/deploy.yml` 会在 `main` 分支推送时运行 `pnpm build` 并发布 Pages。Astro 在 `GITHUB_ACTIONS` 环境下不启用服务端适配器，产出纯静态站点。

### Vercel

导入仓库即可使用 `vercel.json`：安装命令为 `pnpm install`，构建命令为 `pnpm build`。该配置还为通用页面添加基础安全响应头，并为静态资源添加长期缓存。

### Cloudflare Pages / Netlify / EdgeOne

`astro.config.mjs` 会根据 `CF_PAGES`、`NETLIFY` 或 `EDGEONE` 环境变量选择适配器。各平台的构建命令均为 `pnpm build`，输出目录通常为 `dist`。

### Docker

```bash
docker compose up --build -d
```

服务仅绑定到本机 `127.0.0.1:8070`，适合再由宿主机 Nginx、Caddy 或其他反向代理对外提供 HTTPS。Docker 镜像使用多阶段构建，最终由 Nginx 提供静态文件。

## 环境变量

参见 `.env.example`。当前代码识别以下变量：

| 变量 | 用途 |
| --- | --- |
| `OAUTH_GITHUB_CLIENT_ID` | Decap CMS GitHub OAuth Client ID |
| `OAUTH_GITHUB_CLIENT_SECRET` | Decap CMS GitHub OAuth Secret |
| `UMAMI_API_KEY` | Umami API 密钥 |
| `UMAMI_TRACKING_CODE` | Umami 注入代码 |
| `GITHUB_ACTIONS` / `CF_PAGES` / `NETLIFY` / `EDGEONE` / `VERCEL` | 自动选择构建适配器与搜索索引目录 |

## 质量检查与注意事项

提交前建议执行：

```bash
pnpm check
pnpm build
```

- 相册与文章图片数量较多，首次生产构建需要进行 Sharp 图片优化，耗时和磁盘占用会明显高于普通博客。
- 翻译插件包含动态执行逻辑，Vite 会提示 `eval` 风险；本站当前关闭翻译功能。若准备开启，应先审查或替换该插件。
- 音乐、评论、Google Fonts、Iconify 回退、二维码分享与部分友链头像依赖外部网络服务。
- 本地音乐的 `url` / `lrc` 必须能在 `public/` 下找到；没有歌词时请将 `lrc` 留空，避免产生 404 请求。
- `dist/`、`.astro/`、`.vercel/` 和 `node_modules/` 均为生成目录，不应提交。

## License

代码沿用 Twilight 项目的 [MIT License](./LICENSE)。文章、图片、音频等内容资产不因代码许可证而自动获得相同授权；转载或复用前请联系内容作者。
