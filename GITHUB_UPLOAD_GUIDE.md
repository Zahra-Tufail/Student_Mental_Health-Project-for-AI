# 🚀 GitHub Upload Guide

## Safety Checklist Before Uploading

✅ **Never commit sensitive files:**
- `.env` ← API keys live here (NEVER upload!)
- `.env.local` ← Local overrides
- Passwords or secrets

✅ **Files that SHOULD be uploaded:**
- `.env.example` ← Template for others
- `.gitignore` ← Tells Git what to ignore
- All `.py` files
- `requirements.txt`
- CSV files
- Documentation

---

## Step-by-Step: Upload to GitHub

### 1. Initialize Git (if not already done)
```bash
cd e:\ai-project-main
git init
```

### 2. Verify .gitignore Exists
```bash
# Check if .gitignore exists and has .env
type .gitignore
```

You should see `.env` in the output.

### 3. Check What Will Be Uploaded
```bash
git status
```

**Should NOT see:**
- `.env` file
- `__pycache__` folders
- `.streamlit` folder
- `ai/` folder (virtual environment)

**Should see:**
- `app.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- CSV files
- Documentation files

### 4. Add Files to Git
```bash
git add .
```

### 5. Create First Commit
```bash
git commit -m "Initial commit: Student Mental Health Analytics Dashboard"
```

### 6. Create Repository on GitHub
1. Go to [github.com](https://github.com)
2. Click "New repository"
3. Name it: `student-mental-health-analytics`
4. Do NOT initialize with README (we have one)
5. Click "Create repository"

### 7. Connect Local to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/student-mental-health-analytics.git
git branch -M main
git push -u origin main
```

### 8. Verify Upload
Visit your GitHub repo and check:
- ✅ `app.py` uploaded
- ✅ `requirements.txt` uploaded
- ✅ `.env.example` uploaded
- ✅ `.gitignore` shows (confirms `.env` is hidden)
- ❌ `.env` NOT visible
- ❌ `ai/` folder NOT visible

---

## How Others Will Use Your Project

```bash
# 1. Clone from GitHub
git clone https://github.com/YOUR_USERNAME/student-mental-health-analytics.git
cd student-mental-health-analytics

# 2. Setup environment (they see .env.example)
cp .env.example .env

# 3. Add their own API key
# Edit .env with GROQ_API_KEY=their_key

# 4. Run the app
streamlit run app.py
```

---

## Files to Check Before Upload

### ✅ Must Include
```
.env.example          # Template
.gitignore            # Tells Git what to hide
app.py                # Main app
rag/                  # Module
requirements.txt      # Dependencies
*.csv                 # Data
README.md             # Docs
```

### ❌ Must NOT Include
```
.env                  # YOUR API KEY (keep private!)
ai/                   # Virtual environment (too large)
__pycache__/          # Python cache
.streamlit/           # Local config
*.pyc                 # Compiled Python
```

---

## Troubleshooting

**Q: I already committed `.env`!**
```bash
# Remove from Git history
git rm --cached .env
git commit -m "Remove .env file"
git push

# Then regenerate your API key
# It was likely exposed
```

**Q: Should I commit virtual environment?**
No! It's 500MB+ and everyone should create their own.
That's why `.gitignore` excludes `ai/` folder.

**Q: How do I keep secrets safe?**
- Use `.env.example` as template
- Never commit actual `.env`
- Use environment variable names in code
- Load from `.env` only in local development

**Q: What if someone sees my API key in git history?**
1. Regenerate it immediately
2. Force push to rewrite history: `git push --force-with-lease`
3. GitHub shows history, but key will be invalid

---

## Sample .gitignore Setup

Your project includes:

```
.env                    # ← Hide this (your secret key)
.env.local              # ← Hide local overrides
.env.*.local            # ← Hide environment-specific
__pycache__/            # ← Hide Python cache
ai/                     # ← Hide virtual environment
.vscode/                # ← Hide IDE config
.streamlit/             # ← Hide Streamlit cache
```

---

## Final Security Checklist

Before pushing to GitHub:

- [ ] `.env` exists locally (not committed)
- [ ] `.env.example` exists (no real values)
- [ ] `.gitignore` has `.env` added
- [ ] `requirements.txt` is up to date
- [ ] No API keys in any `.py` files
- [ ] No passwords in comments
- [ ] README mentions `.env.example` setup

---

## Links

- GitHub: [github.com](https://github.com)
- Git Docs: [git-scm.com](https://git-scm.com/doc)
- Ignore Templates: [gitignore.io](https://gitignore.io)

**Your project is now ready for the world!** 🌍
