### 2.5D 安检可视化预警系统 - 部署文档

本项目为前端（Vue 3 + Vite + TypeScript）实现的 2.5D 安检可视化预警系统，支持：
- 上传大图，前端自动切片为 Deep Zoom（DZI）瓦片
- OpenSeadragon 多级缩放查看
- 设备标签管理、预警闪烁、事件详情与联动策略配置

提供两种部署方式：
- 方式 A：构建静态前端并由 Nginx（或任意静态服务器）托管
- 方式 B：使用 Node.js/Express 简单后端托管前端静态产物
- 方式 C：Docker 一键构建和运行

---

#### 一、环境要求
- Node.js >= 18（推荐 20）
- npm >= 9（推荐 10+）
- 可选：Docker >= 24

#### 二、本地开发
```
cd safesight-2_5d
npm i
npm run dev
```
默认在 `http://localhost:5173` 访问。

#### 三、生产构建（前端产物）
```
cd safesight-2_5d
npm run build
```
构建产物输出到 `dist/` 目录，可直接由 Nginx/Apache/OSS/CDN 托管。

##### 使用 Nginx 托管（示例）
- 将 `dist/` 上传至服务器目录（如 `/var/www/safesight`）
- 使用仓库内 `nginx.conf` 作为站点配置（或合并至你的 Nginx 配置），确保 SPA 回退到 `index.html`：
```
server {
  listen 80;
  server_name your-domain;
  root /usr/share/nginx/html;

  location / {
    try_files $uri $uri/ /index.html;
  }
}
```

#### 四、使用 Node.js/Express 简单后端托管
```
# 1. 先构建前端
cd safesight-2_5d
npm run build

# 2. 安装并启动后端（仅静态托管 dist）
cd server
npm i
npm start
```
默认在 `http://localhost:8080` 访问（可在 `server/index.js` 修改端口）。

#### 五、Docker 部署
```
# 在项目根目录（包含 Dockerfile 和 nginx.conf）
docker build -t safesight-2_5d:latest .
docker run --rm -p 8080:80 safesight-2_5d:latest
```
随后访问 `http://localhost:8080`。

#### 六、目录说明
- `src/`：前端源码
- `dist/`：前端构建产物（运行 npm run build 后出现）
- `server/`：最简后端（Express），仅托管 `dist/`
- `Dockerfile`：生产镜像（Nginx 托管构建产物）
- `nginx.conf`：Nginx 站点配置（SPA 回退）
- `DEPLOY.md`：本文档

#### 七、已知限制与建议
- 大图在浏览器内进行切片，超大图时建议改为服务端离线切片并静态分发瓦片
- 当前“2.5D”为 2D 底图 + 标注 + 多级缩放，若需倾斜/高度效果，请扩展样式或引入 WebGL 图层