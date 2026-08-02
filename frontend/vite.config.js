import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const DEV_TUNNEL_HOSTS = [".asse.devtunnels.ms"];

function apiProxy(target) {
  return {
    target,
    changeOrigin: true,
    secure: false,
    timeout: 0,
    proxyTimeout: 0,
    rewrite: (path) => path.replace(/^\/api(?=\/|$)/, "")
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.ARAGBIZ_DEV_API_TARGET || "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      strictPort: true,
      allowedHosts: DEV_TUNNEL_HOSTS,
      proxy: {
        "/api": apiProxy(apiTarget)
      }
    },
    preview: {
      host: "127.0.0.1",
      strictPort: true,
      allowedHosts: DEV_TUNNEL_HOSTS,
      proxy: {
        "/api": apiProxy(apiTarget)
      }
    }
  };
});
