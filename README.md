# ScoreSense (local)

ScoreSense reads handwritten school answer booklets and helps teachers mark them. It runs entirely on the school's own computer. The AI model runs locally through [Ollama](https://ollama.com), so student papers never leave the machine.

**What it does today:**

- **Upload an answer booklet** (PDF or photos). The pages are turned upright, each answer is cut out, and answers are grouped by the question number the student wrote in the margin. Answers that run over several pages stay together. Rough-work columns are left out.
- **Flags anything uncertain** for the teacher instead of guessing: unreadable or covered question numbers, repeated numbers, and unreadable marks in the margin.
- **Exam setup:** upload the question paper. The marks scheme is read from its general instructions and checked against the paper's maximum marks. The teacher then clicks the correct option for each multiple-choice question.
- **Marking:** the teacher gives marks per question with buttons or the keyboard, sees a running total, and downloads the marks as a CSV that opens in Excel.

## What you need

| | Minimum | Recommended |
|---|---|---|
| Computer | Windows 10/11, macOS 12+, or Linux | Windows 11 |
| Memory (RAM) | 16 GB | 16 GB or more |
| Graphics card | none (runs on the processor, but slowly: several minutes per page) | NVIDIA with 6 GB or more |
| Disk space | 10 GB free | |
| Internet | only for the first setup | |

On the reference machine (RTX 3050 6 GB, 16 GB RAM), a 24-page booklet takes about 10–15 minutes.

## Install on a new computer

### Windows

1. Install **Python 3.11 or newer** from <https://www.python.org/downloads/>. On the first installer screen, tick **"Add python.exe to PATH"**.
2. Install **Ollama** from <https://ollama.com/download> and start it. It runs in the system tray.
3. Download this project. On GitHub, either click **Code → Download ZIP** and unzip it, or run `git clone <repository URL>`.
4. Double-click **`setup.bat`** in the project folder. It installs the Python packages and downloads the AI model (about 6 GB, so it takes a while the first time).
5. Double-click **`ScoreSense_App.bat`**. Your browser opens **<http://localhost:8000>**. Keep the black window open while you use ScoreSense, and close it to stop.

### macOS / Linux

1. Install Python 3.11+ and Ollama (<https://ollama.com/download>), and start Ollama.
2. Download or clone this project, then in a terminal in the project folder run:

   ```sh
   sh setup.sh      # once
   sh start.sh      # every time; then open http://localhost:8000
   ```

## Using it

1. **Set up the exam (once per exam).** On the home page, click **Set up exam** and upload the question paper.
   - Check the sections and marks. A green tick means they add up to the paper's maximum marks.
   - Click the correct option (A–D) for each multiple-choice question.
2. **Upload a student's answer booklet.** Click the upload box and choose the PDF or photos, with photos in page order. Answers appear page by page while the rest is read.
3. **Mark it.**
   - Type the student's name or roll number.
   - Give each question its marks with the buttons, or type a mark.
   - Keyboard: `0`–`9` gives that mark and moves on, `H` adds half a mark, `Backspace` clears.
   - Look at the orange **need a look** questions first.
4. Click **Download marks** to get the CSV.

## Where the data is kept

Uploaded papers, crops and marks are stored outside the project folder:

- Windows: `C:\ocr_eval\app_jobs\`
- macOS / Linux: `~/scoresense_data/`

To use another folder, set the environment variable `SCORESENSE_DATA` to it before starting. Do not put it inside OneDrive, Dropbox or another synced folder. Student data should stay on the school computer, and none of it is ever committed to this repository (see `.gitignore`).

The app only accepts connections from the computer it runs on (`127.0.0.1`), so other computers on the network cannot open it.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Ollama is not running" or the page stays on the first step | Start the Ollama app, then upload again. |
| `setup.bat` says Python is not installed | Reinstall Python and tick "Add python.exe to PATH". |
| The browser does not open | Open <http://localhost:8000> yourself. |
| Port 8000 is in use | Close the other ScoreSense window, or change `PORT` in `eval/app.py`. |
| Very slow (minutes per page) | No usable graphics card was found, so the model runs on the processor. It still works. |
| A page says "this paper was stopped before it finished" | ScoreSense was closed while reading that paper. Upload it again. |

## Project layout

```
ScoreSense_App.bat, start.sh    start the app
setup.bat, setup.sh             one-time install
eval/app.py                     the web app (FastAPI), pages in eval/app_pages/
eval/booklet.py                 booklet -> upright pages -> answer crops -> question numbers
eval/segment.py                 finds the margin and cuts pages into answers
eval/qpaper.py                  reads the marks scheme from a question paper
eval/model.py                   the local model and the answer-reading prompt
eval/bench_booklet.py           compares models on question-number accuracy
eval/bench_pages.py, score.py   evaluation tools (accuracy of reading)
eval/run.py, ScoreSense.bat     older pipeline for fixed-layout answer sheets (regions.html, mark.py, review.html)
api/, core/, engines/, main.py  earlier experiments with other OCR engines (separate requirements.txt)
```

## Known limits

- **Covered numbers:** a question number hidden under correction fluid leaves no mark on the scan. That answer stays inside the crop of the answer above it.
- **Booklet layout:** booklets need a red or pink margin line on the left. Pages without one (maps, diagrams) are shown whole.
- **Marking is the teacher's:** marks are not given automatically yet. Automatic MCQ checking against the answer key is the next step. Long answers need a stronger model than the local one to suggest marks.
- **One exam at a time:** setting up a new exam replaces the previous one.
