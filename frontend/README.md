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

The frontend expects the FastAPI backend at `http://127.0.0.1:8000`.
Override it with:

```powershell
$env:VITE_ARAGBIZ_API_URL="http://127.0.0.1:8000"
```

If PowerShell cannot find `npm`, prepend the Node.js install folder for the current terminal:

```powershell
$env:Path = "C:\Program Files\nodejs;$env:Path"
npm.cmd install
npm.cmd run dev
```
