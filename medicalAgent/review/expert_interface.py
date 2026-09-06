"""
专家审核接口
支持命令行交互和 REST API（可选）
"""
import json
from typing import List, Dict, Optional
from loguru import logger

from .review_queue import ReviewQueue, ReviewItem, get_default_queue


class ExpertInterface:
    """专家审核接口"""

    def __init__(self, review_queue: Optional[ReviewQueue] = None):
        """
        初始化专家审核接口

        Args:
            review_queue: 审核队列实例（可选，默认使用进程级单例，
                          保证与提问方共享实时审核事件）
        """
        self.queue = review_queue or get_default_queue()

    def list_pending(self) -> List[ReviewItem]:
        """列出待审核项"""
        return self.queue.get_pending("pending")

    def list_all(self, status: Optional[str] = None) -> List[ReviewItem]:
        """列出所有审核项（可按状态过滤）"""
        if status:
            return self.queue.get_pending(status)
        return list(self.queue._items.values())

    def get_detail(self, review_id: str) -> Optional[Dict]:
        """获取审核项详情"""
        item = self.queue.get_item(review_id)
        if not item:
            return None
        return {
            "id": item.id,
            "question": item.question,
            "original_answer": item.original_answer,
            "trigger_reason": item.trigger_reason,
            "review_mode": item.review_mode,
            "status": item.status,
            "expert_correction": item.expert_correction,
            "expert_id": item.expert_id,
            "created_at": item.created_at,
            "reviewed_at": item.reviewed_at,
        }

    def review(
        self,
        review_id: str,
        action: str,
        correction: Optional[str] = None,
        expert_id: str = "anonymous",
    ) -> bool:
        """
        审核操作

        Args:
            review_id: 审核项 ID
            action: "approve" / "correct" / "reject"
            correction: 修正内容（action="correct" 时必填）
            expert_id: 专家 ID

        Returns:
            是否成功
        """
        if action == "approve":
            return self.queue.submit_approval(review_id, expert_id)
        elif action == "correct":
            if not correction:
                logger.error("修正操作需要提供 correction 参数")
                return False
            return self.queue.submit_correction(review_id, correction, expert_id)
        elif action == "reject":
            return self.queue.submit_rejection(review_id, expert_id)
        else:
            logger.error(f"未知操作: {action}")
            return False

    def get_stats(self) -> Dict:
        """获取审核统计"""
        return self.queue.get_stats()

    def run_cli(self):
        """运行命令行交互界面"""
        print("\n" + "=" * 60)
        print("  MediX 专家审核系统")
        print("=" * 60)

        while True:
            pending = self.list_pending()
            stats = self.get_stats()

            print(f"\n待审核: {stats['pending']} | 已批准: {stats['approved']} | "
                  f"已修正: {stats['corrected']} | 已拒绝: {stats['rejected']}")

            if not pending:
                print("\n没有待审核的项目。")
                choice = input("输入 q 退出，或按回车刷新: ").strip()
                if choice.lower() == "q":
                    break
                continue

            # 显示待审核列表
            for i, item in enumerate(pending):
                print(f"\n[{i+1}] ID={item.id} | 原因={item.trigger_reason} | 模式={item.review_mode}")
                print(f"    问题: {item.question[:60]}...")
                print(f"    回答: {item.original_answer[:80]}...")

            print(f"\n操作: <编号> 查看 | a <ID> 批准 | c <ID> <修正> 修正 | r <ID> 拒绝 | q 退出")
            cmd = input("> ").strip()

            if cmd.lower() == "q":
                break
            elif cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(pending):
                    item = pending[idx]
                    detail = self.get_detail(item.id)
                    print(f"\n--- 审核详情 [{item.id}] ---")
                    print(f"问题: {detail['question']}")
                    print(f"\n原始回答:\n{detail['original_answer']}")
                    print(f"\n触发原因: {detail['trigger_reason']}")
                else:
                    print("编号超出范围")
            elif cmd.startswith("a "):
                parts = cmd.split()
                if len(parts) >= 2:
                    rid = parts[1]
                    ok = self.review(rid, "approve")
                    print("已批准" if ok else "操作失败")
            elif cmd.startswith("c "):
                parts = cmd.split(maxsplit=2)
                if len(parts) >= 3:
                    rid, correction = parts[1], parts[2]
                    ok = self.review(rid, "correct", correction=correction)
                    print("已修正" if ok else "操作失败")
                else:
                    print("用法: c <ID> <修正内容>")
            elif cmd.startswith("r "):
                parts = cmd.split()
                if len(parts) >= 2:
                    rid = parts[1]
                    ok = self.review(rid, "reject")
                    print("已拒绝" if ok else "操作失败")
            else:
                print("未知命令")

        print("审核系统已退出。")
