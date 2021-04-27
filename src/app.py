import os
from flask import Flask, request, render_template, send_from_directory

from src.Object_Seg import  frames_to_video

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'files')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for file in request.files.getlist("file"):
        print(file)
        print("{} is the file name".format(file.filename))
        filename = file.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if ext == ".mp4":
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        file.save(destination)
        pathout="/".join([target, "Processed_{}.mp4".format(filename)])
        print(pathout)
        frames_to_video(destination, pathout)

    return send_from_directory("files", "Processed_{}.mp4".format(filename), as_attachment=True)
    #return render_template("complete.html")




if __name__ == "__main__":
    app.run(debug=True)
