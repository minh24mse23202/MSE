# Adaptive RAG Studio Frontend

React/Vite frontend for the Adaptive RAG System.

The current theme follows the supplied `Screen_Login.mhtml` template: dark onboarding background, near-black top navigation, neutral text, and orange primary actions.
The top menu includes a dark/light mode switch; the selected mode is saved in browser local storage.

## Screens

- Splash
- Login
- Sign up
- Main: Conversation History, Chat, RAG Configuration
- Popup Source and Popup Trace from assistant messages
- Knowledge Bases
- Evaluation: Dataset and Evaluation panels
- Evaluation Detail: RAGXplain
- Analytics: Token statistics and Detailed Statistics & Feedbacks tabs

## Run

```powershell
cd frontend
npm install
npm run dev
```

The frontend uses the same-origin `/api` path. Vite proxies that path to
`http://127.0.0.1:8000` during development and preview. To change only the
local proxy target, create `frontend/.env.local` with:

```powershell
ARAGBIZ_DEV_API_TARGET=http://127.0.0.1:8000
```

`VITE_ARAGBIZ_API_URL` is optional and should be used only when the browser is
intended to call a separately exposed API origin. Leave it unset when using a
single Microsoft Dev Tunnel for port `5173`.

## Microsoft Dev Tunnel demo

Start FastAPI on port `8000`, start the background worker, and then run the
Vite development server. Set only port `5173` to **Public**. Verify the proxy
through the public URL before logging in:

```text
https://<tunnel>-5173.asse.devtunnels.ms/api/health
```

The browser should send login, streaming chat, trace, and RAGXplain requests
to `/api/*`; it should not contain requests to `127.0.0.1:8000`.

If PowerShell cannot find `npm`, prepend the Node.js install folder for the current terminal:

```powershell
$env:Path = "C:\Program Files\nodejs;$env:Path"
npm.cmd install
npm.cmd run dev
```
