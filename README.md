# LUMHS AI Chatbot

A fully free, locally hosted AI chatbot for Liaquat University of Medical & Health Sciences (LUMHS), Jamshoro. Answers student and visitor queries using data scraped directly from the university website. No paid APIs, no cloud costs — runs entirely on the university's own server.

---

## How It Works

```
User asks question
       ↓
FastAPI server receives it
       ↓
ChromaDB searches 12,000+ chunks from LUMHS website
       ↓
Relevant data sent to local Ollama AI model
       ↓
Natural answer returned to user
```

---

## Project Structure

```
uni-chatbot/
├── scraper.py          # Scrapes all pages + PDFs from lumhs.edu.pk
├── embed.py            # Chunks scraped data and stores in ChromaDB
├── server.py           # FastAPI server — handles chat requests
├── update.py           # Runs scraper + embedder together (manual update)
├── chat.py             # Terminal test client
├── query.py            # Direct ChromaDB search test
├── widget.html         # Chat bubble UI for embedding on website
├── scraped_data.json   # Output of scraper (auto-generated)
└── chroma_db/          # Vector database (auto-generated)
```

---

## Requirements

### System Requirements
- Python 3.10 or higher
- Google Chrome (for Selenium scraper)
- 4GB+ free RAM (8GB+ recommended for server)
- Windows / Linux / macOS

### Ollama (Local AI)
Ollama runs the AI model locally — no API cost, no internet needed after setup.

**Install Ollama:**

Windows:
```
Download from https://ollama.com and run installer
```

Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Pull AI model:**

For laptop (low RAM):
```bash
ollama pull llama3.2:1b
```

For server with 8GB RAM:
```bash
ollama pull llama3.1:8b
```

For server with 16GB+ RAM:
```bash
ollama pull gemma4:12b
```

**Test Ollama is working:**
```bash
ollama run llama3.2:1b
```
Type `hello` — if it responds, it works. Type `/bye` to exit.

---

## Installation

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/yourusername/lumhs-chatbot.git
cd lumhs-chatbot
```

**Step 2 — Install Python dependencies:**
```bash
pip install requests beautifulsoup4 chromadb sentence-transformers fastapi uvicorn gunicorn httpx selenium webdriver-manager pymupdf psutil
```

**Step 3 — Install Chrome (for scraper):**

Linux server:
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb -y
```

Windows: Chrome is likely already installed. If not, download from google.com/chrome.

---

## First Time Setup

Run these in order — only needed once:

**Step 1 — Scrape the website:**
```bash
python scraper.py
```
This visits every page and PDF on lumhs.edu.pk and saves to `scraped_data.json`.
Takes 1-2 hours. Do not interrupt.

**Step 2 — Build the vector database:**
```bash
python embed.py
```
Reads `scraped_data.json`, creates chunks, stores in ChromaDB.
Takes 20-40 minutes depending on hardware.

**Step 3 — Start the server:**
```bash
python server.py
```
Server starts at `http://0.0.0.0:5000`. Keep this running.

**Step 4 — Test it works:**
```bash
python chat.py "what faculties are in LUMHS"
python chat.py "how to apply for MBBS admission"
python chat.py "what is the contact number of LUMHS"
```

---

## Running the Chat Widget

Open `widget.html` in your browser. Make sure `server.py` is running first.

The chat bubble appears in the bottom right corner. Click it to open.

---

## Updating Data (Manual)

When LUMHS updates their website, run:
```bash
python update.py
```
This clears old data, re-scrapes, and re-embeds everything automatically.

---

## Production Deployment (Linux Server)

### 1. Change model in server.py
```python
OLLAMA_MODEL = "llama3.1:8b"  # or gemma4:12b
```

### 2. Run multiple Ollama instances (for high traffic)
Open 3 terminals:
```bash
# Terminal 1
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Terminal 2
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 3
OLLAMA_HOST=0.0.0.0:11436 ollama serve
```

Update `OLLAMA_URLS` in server.py:
```python
OLLAMA_URLS = [
    "http://localhost:11434/api/generate",
    "http://localhost:11435/api/generate",
    "http://localhost:11436/api/generate",
]
```

### 3. Run server with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:5000
```

### 4. Make server run forever (systemd)
```bash
sudo nano /etc/systemd/system/lumhs.service
```
Paste:
```
[Unit]
Description=LUMHS Chatbot Server
After=network.target

[Service]
WorkingDirectory=/path/to/uni-chatbot
ExecStart=gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:5000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable lumhs
sudo systemctl start lumhs
```

### 5. Daily auto-update (cron job)
```bash
crontab -e
```
Add:
```bash
0 2 * * * cd /path/to/uni-chatbot && python3 update.py >> update.log 2>&1
```
Runs every night at 2am automatically.

### 6. Install Nginx
```bash
sudo apt install nginx -y
sudo nano /etc/nginx/sites-available/lumhs
```
Paste:
```nginx
server {
    listen 80;
    server_name YOUR_SERVER_IP;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
        add_header Access-Control-Allow-Origin "https://www.lumhs.edu.pk";
        add_header Access-Control-Allow-Methods "POST, GET, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type";
    }

    location /widget.html {
        root /path/to/uni-chatbot;
    }
}
```
Then:
```bash
sudo ln -s /etc/nginx/sites-available/lumhs /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

---

## Embedding Widget on LUMHS Website

Update this line in `widget.html`:
```javascript
const LUMHS_API = 'http://127.0.0.1:5000/ask';
```
To:
```javascript
const LUMHS_API = 'http://YOUR_SERVER_IP/ask';
```

Then give IT team this one line to add before `</body>` on every page:
```html
<iframe src="http://YOUR_SERVER_IP/widget.html" style="position:fixed;bottom:0;right:0;width:420px;height:600px;border:none;z-index:9999;"></iframe>
```

---

## Useful Commands

Check if server is running:
```bash
sudo systemctl status lumhs
```

Restart server:
```bash
sudo systemctl restart lumhs
```

Check update logs:
```bash
cat update.log
```

Check ChromaDB chunk count:
```bash
python -c "import chromadb; c = chromadb.PersistentClient('./chroma_db'); col = c.get_collection('lumhs'); print('Chunks:', col.count())"
```

Clear ChromaDB (before re-embedding):
```bash
python -c "import shutil; shutil.rmtree('./chroma_db'); print('Cleared')"
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Scraper | Python + Selenium + BeautifulSoup + PyMuPDF | Crawls website + PDFs |
| Vector DB | ChromaDB | Stores and searches text chunks |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Converts text to vectors |
| AI Model | Ollama (llama3.1:8b / gemma4:12b) | Generates answers |
| API Server | FastAPI + Uvicorn | Handles chat requests |
| Production | Gunicorn + Nginx | Scaling and load balancing |
| Frontend | Vanilla HTML/CSS/JS | Chat widget |

---

## Troubleshooting

**Server won't start:**
```bash
# Check if port 5000 is already in use
netstat -tulnp | grep 5000
```

**Ollama not responding:**
```bash
ollama list        # check model is installed
ollama serve       # start ollama manually
```

**ChromaDB corrupted:**
```bash
python -c "import shutil; shutil.rmtree('./chroma_db'); print('Cleared')"
python embed.py    # re-embed
```

**Scraper getting 0 pages:**
```bash
python -c "import requests; r = requests.get('https://www.lumhs.edu.pk'); print(r.status_code)"
# Should print 200
```

---

## License

Built for LUMHS — Liaquat University of Medical & Health Sciences, Jamshoro, Pakistan.
