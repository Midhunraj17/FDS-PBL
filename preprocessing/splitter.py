def chronological_split(data, train=0.70, validation=0.15):
    n=len(data); a=int(n*train); b=int(n*(train+validation))
    return data[:a], data[a:b], data[b:]
