import asyncio
import os
import re
from kubernetes_asyncio import client, config

async def delete_all_llm_deployments(apps_api, namespace="allworkers"):
    deployments = await apps_api.list_namespaced_deployment(namespace=namespace)
    pattern = re.compile(r"^llm(\d+)-worker(\d+)$")
    # pattern = re.compile(r"^prepull-llm$")
    for dep in deployments.items:
        if pattern.match(dep.metadata.name):
            print(f"Deleting Deployment: {dep.metadata.name}")
            try:
                await apps_api.delete_namespaced_deployment(
                    name=dep.metadata.name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(grace_period_seconds=0, propagation_policy='Foreground')
                )
            except client.ApiException as e:
                if e.status != 404:
                    print(f"Failed to delete deployment {dep.metadata.name}: {e}")

async def create_prepull_daemonset(apps_api, namespace="allworkers"):
    image_prefix = os.getenv("IMAGE_PREFIX", "local/elm")
    container_list = [
        client.V1Container(
            name="prepull-llm1",
            image=f"{image_prefix}-lm1",
            command=["sleep", "3600"],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "10m", "memory": "32Mi"},
                limits={"cpu": "10m", "memory": "32Mi"}
            ),
            image_pull_policy="Always"
        ),
        client.V1Container(
            name="prepull-llm2",
            image=f"{image_prefix}-lm2",
            command=["sleep", "3600"],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "10m", "memory": "32Mi"},
                limits={"cpu": "10m", "memory": "32Mi"}
            ),
            image_pull_policy="Always"
        ),
        client.V1Container(
            name="prepull-llm3",
            image=f"{image_prefix}-lm3",
            command=["sleep", "3600"],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "10m", "memory": "32Mi"},
                limits={"cpu": "10m", "memory": "32Mi"}
            ),
            image_pull_policy="Always"
        ),
        client.V1Container(
            name="prepull-llm4",
            image=f"{image_prefix}-lm4",
            command=["sleep", "3600"],
            resources=client.V1ResourceRequirements(
                requests={"cpu": "10m", "memory": "32Mi"},
                limits={"cpu": "10m", "memory": "32Mi"}
            ),
            image_pull_policy="Always"
        )
    ]

    ds = client.V1DaemonSet(
        api_version="apps/v1",
        kind="DaemonSet",
        metadata=client.V1ObjectMeta(name="prepull-llm", namespace=namespace),
        spec=client.V1DaemonSetSpec(
            selector=client.V1LabelSelector(match_labels={"app": "prepull-llm"}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": "prepull-llm"}),
                spec=client.V1PodSpec(
                    service_account_name="custom-sa",  
                    containers=container_list,
                    tolerations=[],
                    node_selector={"dedicated": "worker"},
                    image_pull_secrets=[client.V1LocalObjectReference(name="llm-docker")],
                    termination_grace_period_seconds=5
                )
            ),
            update_strategy=client.V1DaemonSetUpdateStrategy(type="RollingUpdate")
        )
    )

    try:
        await apps_api.create_namespaced_daemon_set(namespace=namespace, body=ds)
        print("DaemonSet prepull-llm created.")
    except client.ApiException as e:
        print(f"Failed to create DaemonSet: {e}")

async def wait_for_daemonset_ready(apps_api, namespace="allworkers", ds_name="prepull-llm", timeout=30000, interval=30):
    """Wait for the DaemonSet to have all pods ready.
    Return True if ready, False if timed out."""
    end_time = asyncio.get_event_loop().time() + timeout
    while True:
        ds = await apps_api.read_namespaced_daemon_set(name=ds_name, namespace=namespace)
        desired = ds.status.desired_number_scheduled
        ready = ds.status.number_ready
        print(f"DaemonSet {ds_name}: desired={desired}, ready={ready}")

        if desired is not None and desired > 0 and desired == ready:
            print("All DaemonSet pods are ready.")
            return True

        if asyncio.get_event_loop().time() > end_time:
            print("Timed out waiting for DaemonSet pods to become ready.")
            return False

        await asyncio.sleep(interval)

async def delete_daemonset(apps_api, namespace="allworkers", ds_name="prepull-llm"):
    try:
        await apps_api.delete_namespaced_daemon_set(
            name=ds_name,
            namespace=namespace,
            body=client.V1DeleteOptions(grace_period_seconds=0, propagation_policy='Foreground')
        )
        print(f"DaemonSet {ds_name} deleted.")
    except client.ApiException as e:
        if e.status != 404:
            print(f"Failed to delete DaemonSet {ds_name}: {e}")

async def main():
    try:
        await config.load_incluster_config()
    except:
        await config.load_kube_config()

    apps_api = client.AppsV1Api()

    # Remove stale LM deployments before a new controller run.
    await delete_all_llm_deployments(apps_api)

    # Create a short-lived DaemonSet to pre-pull LM images on worker nodes.
    await create_prepull_daemonset(apps_api)

    # Wait until image pre-pull is complete.
    success = await wait_for_daemonset_ready(apps_api)
    if not success:
        print("DaemonSet never became ready, not deleting it. Check your images or network.")
        await apps_api.api_client.close()
        return

    await apps_api.api_client.close()

    # # 4. delete DaemonSet only after pods are ready
    # await delete_daemonset(apps_api)
    
if __name__ == "__main__":
    asyncio.run(main())
