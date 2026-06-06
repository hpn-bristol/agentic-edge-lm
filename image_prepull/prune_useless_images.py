import paramiko
import os
import sys

nodes = [h.strip() for h in os.getenv("WORKER_HOSTS", "").split(",") if h.strip()]

image_prefix = os.getenv("IMAGE_PREFIX", "local/elm")
command = f"docker images | grep '{image_prefix}-sadrl' | grep '<none>' | awk '{{print $3}}' | xargs -r docker rmi"

username = os.getenv("WORKER_SSH_USER", "ubuntu")
key_filename = os.path.expanduser(os.getenv("WORKER_SSH_KEY", "~/.ssh/id_rsa"))

def execute_command_on_node(host, user, cmd, key_file=None):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    client.connect(hostname=host, username=user, key_filename=key_file, timeout=10)

    stdin, stdout, stderr = client.exec_command(cmd)

    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    
    client.close()
    return out, err

if __name__ == "__main__":
    if not nodes:
        sys.exit("Set WORKER_HOSTS to a comma-separated list of worker SSH hosts.")
    for node in nodes:
        print(f"Cleaning <none> tagged elm-sadrl images on node: {node}")
        out, err = execute_command_on_node(node, username, command, key_filename)
        
        if out:
            print(f"[{node}] Output:\n{out}")
        if err:
            print(f"[{node}] Error:\n{err}")
        
        print(f"[{node}] Done.\n")
