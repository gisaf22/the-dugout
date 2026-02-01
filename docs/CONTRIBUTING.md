# Contributing

## Branching Workflow

No direct commits to `main`. All changes go through a feature branch.

| Branch | Purpose |
|--------|---------|
| `main` | Stable, deployable |
| `feature/<desc>` | New work |
| `fix/<desc>` | Bug fixes |

### Start a feature

```bash
git checkout main
git pull origin main
git checkout -b feature/free-hit-backtest
```

### Push and merge

```bash
git push -u origin feature/free-hit-backtest
# Open PR on GitHub, then merge to main
```

### After merge

```bash
git checkout main
git pull origin main
git branch -d feature/free-hit-backtest
```

## What Not to Touch

Production decision logic and frozen contracts are off-limits without explicit discussion:

- `src/dugout/production/decisions/`
- `src/dugout/production/models/`
- Decision rule: `argmax(predicted_points)`

## Data Automation

See [config/com.thedugout.pull-fpl-data.plist](../config/com.thedugout.pull-fpl-data.plist) for macOS LaunchAgent setup.

### Install

```bash
mkdir -p ~/Library/Logs/the-dugout
cp config/com.thedugout.pull-fpl-data.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.thedugout.pull-fpl-data.plist
```

### Unload

```bash
launchctl unload ~/Library/LaunchAgents/com.thedugout.pull-fpl-data.plist
```

### Check status

```bash
launchctl list | grep thedugout
```

### View logs

```bash
tail -f ~/Library/Logs/the-dugout/pull_fpl_data.log
tail -f ~/Library/Logs/the-dugout/pull_fpl_data.error.log
```

### Behavior

- Runs on login
- Re-runs every 6 hours
- Waits for network
- Fails silently (no retries)
