#!/usr/bin/env python3
"""
Celery task cleanup script.

This script can be used to:
1. Purge all pending tasks from the Celery queue
2. Revoke all active and scheduled tasks
3. Clean up task results from the result backend

Usage:
    python clear_celery_tasks.py --purge        # 清理队列中的所有任务
    python clear_celery_tasks.py --revoke       # 撤销所有任务
    python clear_celery_tasks.py --all          # 执行所有清理操作
"""

import argparse
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.celery_app import celery_app


def purge_queue():
    """Purge all pending tasks from the Celery queue."""
    print("Purging all pending tasks from the queue...")
    try:
        count = celery_app.control.purge()
        print(f"Purged {count} pending tasks from the queue.")
    except Exception as e:
        print(f"Error purging queue: {e}")


def revoke_all_tasks():
    """Revoke all active and scheduled tasks."""
    print("Revoking all active and scheduled tasks...")
    try:
        # 获取所有worker的active和scheduled任务
        inspect = celery_app.control.inspect()
        
        # 获取active任务
        active_tasks = inspect.active()
        if active_tasks:
            for worker, tasks in active_tasks.items():
                task_ids = [task['id'] for task in tasks]
                celery_app.control.revoke(task_ids, terminate=True)
                print(f"Revoked {len(task_ids)} active tasks from worker {worker}")
        
        # 获取scheduled任务  
        scheduled_tasks = inspect.scheduled()
        if scheduled_tasks:
            for worker, tasks in scheduled_tasks.items():
                task_ids = [task['request']['id'] for task in tasks]
                celery_app.control.revoke(task_ids, terminate=True)
                print(f"Revoked {len(task_ids)} scheduled tasks from worker {worker}")
                
        print("All tasks revoked successfully.")
    except Exception as e:
        print(f"Error revoking tasks: {e}")


def cleanup_results():
    """Clean up task results from the result backend."""
    print("Cleaning up task results...")
    try:
        # 如果使用Redis作为结果后端，可以清理相关键
        from backend.app.config.settings import get_settings
        settings = get_settings()
        
        if 'redis' in settings.CELERY_RESULT_BACKEND.lower():
            import redis
            from urllib.parse import urlparse
            
            parsed = urlparse(settings.CELERY_RESULT_BACKEND)
            r = redis.Redis(
                host=parsed.hostname or 'localhost',
                port=parsed.port or 6379,
                db=int(parsed.path.lstrip('/')) if parsed.path else 1,
                password=parsed.password
            )
            
            # 删除所有以celery-task-meta-开头的键
            keys = r.keys('celery-task-meta-*')
            if keys:
                r.delete(*keys)
                print(f"Deleted {len(keys)} task result keys from Redis.")
            else:
                print("No task result keys found in Redis.")
        else:
            print("Result backend is not Redis, skipping cleanup.")
            
    except Exception as e:
        print(f"Error cleaning up results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Celery Task Cleanup Tool')
    parser.add_argument('--purge', action='store_true', 
                       help='Purge all pending tasks from the queue')
    parser.add_argument('--revoke', action='store_true', 
                       help='Revoke all active and scheduled tasks')
    parser.add_argument('--results', action='store_true', 
                       help='Clean up task results from result backend')
    parser.add_argument('--all', action='store_true', 
                       help='Perform all cleanup operations')
    
    args = parser.parse_args()
    
    if not any([args.purge, args.revoke, args.results, args.all]):
        parser.print_help()
        return
    
    if args.all:
        purge_queue()
        revoke_all_tasks()
        cleanup_results()
    else:
        if args.purge:
            purge_queue()
        if args.revoke:
            revoke_all_tasks()
        if args.results:
            cleanup_results()


if __name__ == '__main__':
    main()