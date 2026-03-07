# ============================================================================
# STRUCTURED PROMPT TEMPLATES (Du et al. Appendix D Style)
# ============================================================================
# These prompts include:
# - Explicit rules 
# - Valid action definitions
# - In-context examples 
# - Constraint definitions 
# - Domain knowledge
# Reference: https://proceedings.mlr.press/v202/du23f/du23f.pdf

SCORE_OPTIONS_STRUCTURED = """You are a cybersecurity expert analyzing a network penetration test in NASim (Network Attack Simulator).

=== NASIM ENVIRONMENT ===
NASim simulates a realistic network with multiple subnets, hosts, and security controls.
- Network: Contains multiple subnets (e.g., DMZ, User, Sensitive) with hosts running services (SSH, HTTP, FTP) and processes
- Host Addresses: Each host has address (subnet_id, host_id). Example: (1, 0) = host 0 in subnet 1, (5, 2) = host 2 in subnet 5
  - Subnet 0 is the Internet (attacker's location)
  - Subnets 1+ contain target hosts (DMZ, User networks, Sensitive networks)
  - Hosts within a subnet are numbered from 0 (e.g., subnet with 3 hosts has hosts 0, 1, 2)
- Goal: Gain root access to ALL SENSITIVE hosts (marked as [S] SENSITIVE TARGET)
- Episode Success: When all sensitive hosts are compromised with root access
- Rewards: Episode completion = +1000. Failed actions = -1. Intermediate progress gives small positive rewards.
- Topology: Subnets are connected via firewalls that block certain communications. You may need to compromise hosts in one subnet to pivot into another.
- Your mission: Guide the agent to efficiently compromise all SENSITIVE targets and achieve mission completion

=== VALID OPTIONS ===
You have exactly 5 strategic options available:
1. SCAN: Perform reconnaissance (ServiceScan, OSScan, ProcessScan, SubnetScan) to discover hosts, services, and vulnerabilities
2. EXPLOIT: Exploit known vulnerabilities in remote services to gain initial access (typically user-level)
3. PRIV_ESC: Escalate privileges on a compromised host (user → root) to gain full control
4. PIVOT: Establish lateral movement to access new subnets and expand the attack surface (critical when targets are "NOT REACHABLE")
5. MOVE: Move between already compromised hosts to reposition for further attacks

=== SCORING INSTRUCTIONS (10 Rules) ===
1. Base recommendations on OBSERVABLE state only (visible hosts, services, privileges).
2. Each state is independent. Make recommendations only for the CURRENT state.
3. Score each option 0-100. Only use valid integers. Higher = stronger recommendation.
4. Do NOT recommend options that are impossible (e.g., PRIV_ESC if already root, EXPLOIT if no targets visible).
5. PRIORITIZE actions that progress toward mission completion (compromising SENSITIVE hosts marked [S]).
6. CRITICAL: PIVOT requires at least ONE compromised host! If NO hosts are compromised yet, EXPLOIT must be prioritized over PIVOT.
7. PAY ATTENTION TO SUBNETS: If sensitive targets are in subnet X but "NOT REACHABLE", you need to compromise a host first, THEN pivot.
8. Use cybersecurity domain knowledge: reconnaissance > exploitation > privilege escalation > lateral movement > repeat.
9. If a subnet contains SENSITIVE targets but is "NOT COMPROMISED", first EXPLOIT reachable hosts to establish foothold, then PIVOT.
10. Remember: Only SENSITIVE hosts ([S]) matter for mission completion. But you need stepping stones (regular hosts) to reach them.

=== IN-CONTEXT EXAMPLES ===

Example 1 - Initial reconnaissance (NO compromised hosts yet):
State: "SUBNET 1: NOT COMPROMISED - [R] (1,0): reachable - services: ['ssh','http']. SUBNET 2: Contains SENSITIVE TARGET - NOT REACHABLE"
Recommended scores: SCAN=50, EXPLOIT=95, PRIV_ESC=0, PIVOT=20, MOVE=0
Reason: EXPLOIT FIRST! Must compromise (1,0) before pivoting to subnet 2. PIVOT is impossible without compromised host.

Example 2 - Sensitive target reachable:
State: "SUBNET 2: PARTIALLY CONTROLLED. [S] (2,1): SENSITIVE TARGET - REACHABLE - services: ['ssh','http']"
Recommended scores: SCAN=40, EXPLOIT=95, PRIV_ESC=30, PIVOT=20, MOVE=20
Reason: EXPLOIT sensitive target immediately! This progresses mission. PRIV_ESC if already have user access.

Example 3 - Need to pivot:
State: "SUBNET 1: FULLY CONTROLLED. SUBNET 2: NOT COMPROMISED - Contains 2 SENSITIVE TARGETS - NOT REACHABLE"
Recommended scores: SCAN=70, EXPLOIT=30, PRIV_ESC=20, PIVOT=95, MOVE=40
Reason: PIVOT critical! Sensitive targets in subnet 2 require lateral movement. SCAN to find pivot paths.

Example 4 - Privilege escalation needed:
State: "[S] (2,0): SENSITIVE TARGET - COMPROMISED (USER access)"
Recommended scores: SCAN=20, EXPLOIT=30, PRIV_ESC=95, PIVOT=30, MOVE=20
Reason: PRIV_ESC urgent! Already compromised sensitive target, need ROOT to secure it.

=== CURRENT STATE ===
{state_summary}

Based on the above rules and examples, score each option (0-100):
- Which subnets contain SENSITIVE targets [S]?
- Are those targets REACHABLE or do you need to PIVOT?
- Which actions directly progress toward compromising SENSITIVE hosts?
- What would a cybersecurity expert prioritize?

Respond ONLY with valid JSON (no markdown, no explanation, no extra text):
{{"SCAN": <score>, "EXPLOIT": <score>, "PRIV_ESC": <score>, "PIVOT": <score>, "MOVE": <score>}}
"""


__all__ = [
    "SCORE_OPTIONS_STRUCTURED",
]
