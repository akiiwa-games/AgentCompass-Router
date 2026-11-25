AgentCompass v0.7.9 fixes the tray's quick-menu editor, which v0.7.8 shipped unusable.

- **The list holds what can be added.** **Edit menu…** used to open with one "all models" row per account, which the menu cannot hold, followed by every model, many of which it refuses (an account outside automatic routing, no quota window of a day or longer). It now lists exactly the models the menu can still take, and follows the menu after each change.
- **Adding answers where you are looking.** The result of an action was written to the panel's status line, behind the editor's window, so a refused choice looked like nothing happened. The editor shows it itself.
- **One button.** **Add to quick menu** is the only add; the expected time to the reset is chosen on a menu entry (1 to 24 hours). The second form, which set a rule for a chosen target and hours, is removed, and with it setting a priority for a whole account from the tray.
- **The first edit keeps the names of the shipped entries.**

The full list is in the [changelog](https://github.com/akiiwa-games/AgentCompass-Router/blob/v0.7.9/CHANGELOG.md).

Verified on the released tree: the full test suite; the foreign-machine check F01 to F13; the zero-dispatch check over 13 surfaces; the published reference data against every released reader of its schema; the editor run end to end in the test world (the real data functions, the real preferences file, the real widgets): add, the entry in the menu, the choice gone from the list, the answer in the window, remove. The Windows Sandbox installer check failed one assertion on the first run, a click the check placed by fixed coordinates that missed after the host's display scaling changed; the check now places it from the screen's edge and passed when run again. Not verified: nobody has clicked through the editor in an installed tray yet.

The Windows tray installer is unsigned; Windows SmartScreen may warn on first launch. See [packaging/SIGNING.md](https://github.com/akiiwa-games/AgentCompass-Router/blob/v0.7.9/packaging/SIGNING.md).

Status: alpha. Accounts come from discovery on your machine; choose which ones take part in the tray.
