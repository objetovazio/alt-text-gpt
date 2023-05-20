import requests
from concurrent.futures import as_completed, ThreadPoolExecutor

def perform_two_post_requests(url1, url2, data1 = [], data2 = []):
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit the two POST requests as asynchronous tasks
        task1 = executor.submit(requests.post, url1, data=data1)
        task2 = executor.submit(requests.post, url2, data=data2)

        # Wait for both tasks to complete
        for future in as_completed([task1, task2]):
            response = future.result()
            # Handle the response as needed
            print('end1')

    # Both requests have been completed, proceed with another function
    print('end2')

perform_two_post_requests('https://www.google.com', 'https://www.bing.com')