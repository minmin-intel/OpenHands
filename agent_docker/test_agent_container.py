import requests
import time

def test_agent_instance(url, payload=None):
    if payload is None:
        payload = {}
    
    proxies = {"http": ""}
    response = requests.post(url, json=payload, proxies=proxies)
    answer = response.json()
    print(f"Response from server: {answer}")
    return answer

def get_status(url):
    proxies = {"http": ""}
    response = requests.get(url, proxies=proxies)
    status = response.json()
    print(f"Status from server: {status}")
    return status

if __name__ == "__main__":
    # Check Docker status
    # url = "http://localhost:8000/docker-status"
    # check_docker_status = get_status(url)
    # print(f"Docker status: {check_docker_status}")
    # time.sleep(2)
    
    # Create the agent without initialization
    url = "http://localhost:8000/create"
    create_response = test_agent_instance(url, payload={"initialize_now": "true"})
    
    # Get the agent ID
    agent_id = create_response['id']
    print(f"Agent ID: {agent_id}")
    
    # # If you want to create and initialize in one step
    # print("\n=== Creating agent with immediate initialization ===")
    # url = "http://localhost:8000/create"
    # create_response = test_agent_instance(url, payload={"initialize_now": True})
    # agent_id = create_response['id']
    # print(f"Agent ID with initialization: {agent_id}")
    
    # # Example of running a task with the created agent
    # url = f"http://localhost:8000/run"
    # payload = {
    #     "controller_id": agent_id,
    #     "message": "write a python script that prints hello world",
    # }
    
    # response = test_agent_instance(url, payload)
    # print(f"Task response: {response}")
