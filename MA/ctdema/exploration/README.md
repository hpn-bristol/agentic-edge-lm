# Macro Exploration Examples

This directory contains example macro policies used for initial exploration.
They are templates,
not cluster-ready configuration files.

Before using them:

1. Replace `<node-ip-*>` with the InternalIP values of your worker nodes.
2. Replace `worker*` keys with your Kubernetes worker names.
3. Adjust `LM_policies` and `node_layout_intent` to match your hardware.

The macro planner consumes `example_policy_*.json` in lexical order for the
first epochs, then switches back to live LLM planning using the accumulated
history.

To point the controller to a different exploration directory:

```bash
export LLM_MACRO_EXPLORATION_DIR=<path-to-exploration-policies>
```
