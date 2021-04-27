import cv2
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


#video_path = r"C:\Users\andre\OneDrive\Escritorio\Materias Essex\Group Project\Object_Detection\day+at+the+park.mp4"
video_path = r"C:\Users\andre\OneDrive\Escritorio\Materias Essex\Group Project\Object_Detection\Outside.mp4"
pathOut = r'C:\Users\andre\OneDrive\Escritorio\Materias Essex\Group Project\Object_Detection\videoimg.mp4'
#video_path = r"C:\Users\andre\OneDrive\Escritorio\Materias Essex\Group Project\Object_Detection\bike.jpg"
def print_progress(iteration, total, suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    print(' |%s| %s%s %s' % ( bar, percents, '%', suffix)),  # write out the bar
    #sys.stdout.flush()  # flush to stdout

#Function for object classiciation for chunks
def extract_frames(video_path, start=-1, end=-1, every=1):
    frames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    np.random.seed(seed=69)
    COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))


    thresh = 0.5  # confidence threshold to show


    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if not capture.isOpened():
        print("Error opening video  file")

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        ret, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            if ret:
                classIds, confs, bbox = net.detect(image, confThreshold=thresh)
                # print(classIds, bbox)
                if len(classIds) != 0:
                    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                        cv2.rectangle(image, box, color=COLORS[classId - 1], thickness=2)
                        cv2.putText(image, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, COLORS[classId - 1], 2)
                        cv2.putText(image, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, COLORS[classId - 1], 2)
                # Display the resulting frame
                # cv2.imshow('Frame', image)
                height, width, layers = image.shape
                size = (width, height)
                frames.append(image)
                #print('%i/%i' % (len(frames), end))

            # Break the loop
            else:
                break

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return frames,size  # and return the count of the images we saved

def frames_to_video(video_path, video_out, every=1, chunk_size=1000):

    frames=[]
    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()  # release the capture straight away

    print(total)
    if (total<=1000) & (total>100):
        chunk_size=100

    if (total<=100) & (total>10):
        chunk_size=10

    if (total <= 10):
        chunk_size = 1

    print(chunk_size)

    frame_chunks = [[i, i + chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total - 1)  # make sure last chunk has correct end frame, also handles case chunk_size < total

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor: #multiprocessing.cpu_count()

        futures = {executor.submit(extract_frames, video_path, f[0], f[1], every): f for f in frame_chunks}  # submit the processes: extract_frames(...)
        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print_progress(i, len(frame_chunks) - 1, suffix='Complete')  # print it's progress

        for future in futures:
            chunk = futures[future]
            try:
                data = future.result()
                frames.append(data[0])
                size=(data[1])
            except Exception as exc:
                print('%r generated an exception: %s' % (chunk, exc))
            else:
                print('%r page is %d bytes' % (chunk, len(data[0])))
        print(len(frames))
        print(size)


    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for fr in frames:
        for img in fr:
            # writing to a image array
            out.write(img)
    out.release()


if __name__ == '__main__':
    # test it
    frames_to_video(video_path, pathOut)
