"""领域适配层（P5）。

换领域只改配置，不改代码。
"""

from agent_base.domain.adapter import DomainAdapter, load_domain

__all__ = ["DomainAdapter", "load_domain"]
