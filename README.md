# 技术笔记

Hugo + [Pico Base](https://themes.gohugo.io/themes/pico-base/)（[Pico CSS](https://picocss.com/)）。文章在 `content/posts/`。公式用 KaTeX。推 `main` 后发布到 GitHub Pages。

## 本机

需要 [Hugo Extended](https://gohugo.io/installation/) ≥ 0.116。

```bash
git clone git@github.com:KhazanGu/KhazanGu.github.io.git
cd KhazanGu.github.io
hugo server
```

打开 http://localhost:1313/ 。

## 写文章

```bash
hugo new content posts/一篇笔记.md
```

把 `draft = true` 改成 `false`。行内公式用 `\(...\)`，块级用 `$$`。

改站点名：`config/_default/languages.zh.toml` 的 `title`。线上：https://khazangu.github.io/ ，见 [DEPLOY.md](DEPLOY.md)。
