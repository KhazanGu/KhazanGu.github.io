# 发布到 GitHub Pages

远程：`git@github.com:KhazanGu/KhazanGu.github.io.git`  
站点：https://khazangu.github.io/

推送到 `main` 后，GitHub Actions 会构建 Hugo 并发布。生产环境的 `baseURL` 由 Actions 自动写成上面的地址。

## 第一次（若尚未开启 Pages）

仓库 **Settings → Pages** 里 Source 选 **GitHub Actions**。打开 **Actions**，等 **Deploy GitHub Pages** 成功。

克隆：

```bash
git clone git@github.com:KhazanGu/KhazanGu.github.io.git
```

## 以后

改文章、推 `main` 即可。本地预览：

```bash
hugo server
```

打开 http://localhost:1313/ 。
