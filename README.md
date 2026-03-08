# Yolov8n_footfall analysis

This project uses a USB camera to detect pedestrians in real time and allows manual drawing of entry/exit lines on the screen to count IN / OUT / TOTAL FLOW. It is suitable for simple pedestrian flow counting scenarios such as community booths, entrances/exits, corridors, or event venues.

## Usage

After the program starts, a camera preview will be displayed.

- Use the left mouse button to click two points to draw a counting line.

- Press `Enter` to start counting.

- Press `R` to redraw.

- Press `q` to exit.

Pressing `q` will stop the program. If `SAVE_VIDEO=True` is enabled, the system will output the annotated footage as `flow_output.mp4`.

## Final Result

![image](https://github.com/Roy6842/Yolov8n_Footfall-Analysis/blob/main/images/img_001.png)

## Reference

https://github.com/Daniel-ShinShen/pythonProject_People_Counting_Yolov8_Deepsort/tree/main

https://github.com/harshaa1231/People_detection
