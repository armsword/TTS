# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
需求详情@spec.md

## 技术方案
技术方案@plan.md

## Tasks
原子任务列表@tasks.md，每个任务完成后都要尽可能详细的输出实现原理

## TDD
- 严格按照TDD测试驱动开发三条规则，每次只写一个失败测试，再写实现

## User
- 完成后每次回复我前，都要用 若飞 称呼我
- 每个功能完成（比如算子怎么实现）后需要编写一份详细的原理文档，放到docs目录里，方便我明白底层实现原理

## Git 工作流
- 每个功能 / 接口完成后立即 commit，不要等到全部写完再提交，按原子任务粒度提交，保证单次提交最小化、完整性
- Commit message 使用中文，格式：`[功能名] 简短描述`
- Commit 前确保测试通过
- commit完成后，通过git push推送到远程分支
