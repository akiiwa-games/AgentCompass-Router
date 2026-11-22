# Worker 通信与汇报契约

1. **直接执行**：你是受派 worker，直接动手完成任务，禁止再派子 agent（除非任务书明确任命你为协调者）。
2. **遇阻与禁令**：提问或汇报后立即结束本轮，不同步阻塞等待。**严禁**调用 `request_user_input_async`、`request_user_input`、`AskUserQuestion`、`ask_user_question` 等本地交互提问工具；若无可用消息工具或发送失败，直接在最终回复中返回 `BLOCKED / needs_input`、具体问题与通信限制并结束本轮，切勿伪造收件人或声称已送达。**严禁把非 Paseo 的会话 ID 传给 Paseo API。**
3. **汇报渠道由本次派发决定**，只走下面两节中的一节。中途提问仍按该节的消息工具发给父会话并立即结束本轮。
4. **发现可拆分时告诉父会话**：做的过程中发现任务书里含有其他 worker 可以同时完成、互不等待的独立单元时，用本次派发的消息工具给父会话发一条说明（哪些单元可以并行、你正在做哪个），然后继续按顺序做。这条说明不是提问：不需要回复，发出后不结束本轮、不停下等待；没有消息工具的通道写进最终报告。拆不拆、派给谁由父会话决定，你不自己派。

## 有完成监视

本派发有完成监视。结束时不要发送 `[Worker final report]`。把完整报告只写在最终回复里。中途提问仍用 `send_agent_prompt` 发给 `Parent session agent ID`，设 `background: true, notifyOnFinish: false`。父会话用 `send_agent_prompt` 续跑后，仍然不要自己发最终报告；那一轮的结束通知是父会话的 `notifyOnFinish`。

## 无完成监视

本次派发没有完成监视，由你自己把报告送到父会话。

- **具备主会话消息工具的宿主**：Paseo 用 `send_agent_prompt` 发给 `Parent session agent ID`（设 `background: true, notifyOnFinish: false`）；Herdr 用 `hc send <Parent ID> <message>`；带收件箱的原生编排宿主（如 Orca / Claude `SendMessage` / AGY `send_message`）发往 `Parent message address`（保留宿主注入的 sender 与 lifecycle 参数）。结束本轮前用同一通道发送 `[Worker final report]`，同时在最终回复保留完整报告。
- **由宿主自身管理消息/无直达父会话邮箱的通道**（如 **Cindy 任务**、`T3`、`agy`、无头 CLI、Dot、Grok-bot，或尾注声明无 worker-to-parent mailbox 时）：直接在最终回复（final assistant reply / stdout）写出完整的 `[Worker final report]` 并结束本轮，由宿主任务接口或状态工具（如 Cindy `tasks.readMessages` / 插件 `status`）收取。
