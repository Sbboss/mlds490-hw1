# What you still have to do

The code, figures, validation, and `report/report.pdf` are in this folder.
Three strings in the report are placeholders. Nothing in the repo invents them.

1. Open `report/report.md` and replace, in the title block:
   - `<<GITHUB_USERNAME>>`
   - `<<REPO_NAME>>`
   - `<<GITHUB_LINK>>`
   Then rebuild the PDF (commands are in the report's "How to run" section, and again at the bottom of this file).

2. Read the AI-assistance paragraph in the report and confirm it matches what you actually did. Edit it if it does not. The draft says the implementation and the runs were produced with Cursor, and that you are responsible for the numbers and for being able to explain the code. `STUDY_GUIDE.md` is the explanation.

3. Create the GitHub repo and push. If `gh auth status` works:

   ```bash
   cd "/Users/sbboss/NU/Q4/MLDS-490/HW#1"
   gh repo create <<REPO_NAME>> --public --source . --push
   ```

   If it does not, the commands are:

   ```bash
   cd "/Users/sbboss/NU/Q4/MLDS-490/HW#1"
   git remote add origin git@github.com:<<GITHUB_USERNAME>>/<<REPO_NAME>>.git
   git push -u origin HEAD
   ```

   Do not commit `.env`, tokens, or SSH keys. None were written into this folder.

4. Submit on Canvas, before Mon 2026-10-26 11:59 pm CT:
   - the PDF (`report/report.pdf`)
   - the GitHub link

5. Optional SSH convenience, only if you want it. This was **not** written into `~/.ssh/config`:

   ```
   Host deepdish
     HostName mlds-deepdish5.ads.northwestern.edu
     User hia9393
     ControlMaster auto
     ControlPath ~/.ssh/cm-%r@%h:%p
     ControlPersist 8h
     ServerAliveInterval 30
   ```

## Rebuild the PDF after you edit the placeholders

The Anaconda `pandoc` on this machine is an x86_64 binary and does not run. Rebuild with:

```bash
cd "/Users/sbboss/NU/Q4/MLDS-490/HW#1"
.venv/bin/python scripts/build_report.py
```

Open `report/report.pdf` and check that every figure is visible and no table runs off the page.
