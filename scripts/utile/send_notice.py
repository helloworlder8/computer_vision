import requests
def send_notice(content):
    token = "853672d072e640479144fba8b29b314b"
    title = "训练成功"
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template=html"
    response = requests.request("GET", url)
    print(response.text)
    
    
def send_notice_by_task(metrics,task):
    if task == "detect":
        
        send_notice(f"metrics/mAP50(B): {metrics.results_dict['metrics/mAP50(B)']}, "
            f"metrics/mAP50-95(B): {metrics.results_dict['metrics/mAP50-95(B)']}, "
            # f"mAP50(B): {metrics.results_dict['metrics/mAP50(M)']}, "
            # f"mAP50-95(B): {metrics.results_dict['metrics/mAP50-95(M)']}, "
            f"Fitness: {metrics.results_dict['fitness']}")
    elif task == "segment":
        send_notice(f"metrics/mAP50(B): {metrics.results_dict['metrics/mAP50(B)']}, "
            f"metrics/mAP50-95(B): {metrics.results_dict['metrics/mAP50-95(B)']}, "
            f"metrics/mAP50(M): {metrics.results_dict['metrics/mAP50(M)']}, "
            f"metrics/mAP50-95(M): {metrics.results_dict['metrics/mAP50-95(M)']}, "
            f"Fitness: {metrics.results_dict['fitness']}") 
    else:
        send_notice("yes")