import json 
import youtube_dl
import sys

# Open JSON file
f = open('WLASL_v0.3.json')

data = json.load(f)

for i in data:
    if "gloss" in i:
        if i["gloss"] == sys.argv[1].lower() and "instances" in i:
            vid_num = 0
            for instance in i["instances"]:
                if instance["frame_end"] == -1:
                    try:
                        ydl_opts = {'outtmpl':
                                'videos/{}/{}'.format(sys.argv[1],
                                    str(vid_num)), 'format':'137'}
                        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([instance["url"]])
                        vid_num += 1
                    except Exception as e:
                        pass

f.close()
