
import requests

def create_agent_instance():
    url = "http://localhost:8000/create"
    proxies = {"http": ""}
    payload = {}
    response = requests.post(url, proxies=proxies)
    answer = response.json()
    print(f"Response from server: {answer}")
    print(f"Agent created with ID: {answer['id']}")
    return answer

if __name__ == "__main__":
    # Create the agent
    create_response = create_agent_instance()
    
    # Now you can use the agent ID from the response to run tasks
    # agent_id = create_response['id']
    # print(f"Agent ID: {agent_id}")
    
    # # Example of running a task with the created agent
    # url = "http://localhost:8000/process"
    # proxies = {"http": ""}
    # payload = {
    #     "task": "example_task",
    #     "input": "example_input"
    # }
    
    # response = requests.post(url, json=payload, proxies=proxies)
    # task_status = response.json()
    # print(f"Task status: {task_status}")
