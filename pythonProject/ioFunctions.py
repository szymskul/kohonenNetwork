def writeStats(fileName, stat):
    with open(fileName, 'a') as file:
        file.write(str(stat))
        file.write('\n')