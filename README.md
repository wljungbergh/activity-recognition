# Project in autonomous systems WASP course 2022
## Actvity recognition
We should be able to distinguish between standing still, walking and running given sensor data available on a smartphone. The solution used a sliding window in combination with a Fast Fourier Transform (FFT) on the accelerometer data in order to distinguish between the different modes.

If you want to run our solution you need only a logfile collected with the SensorFusion app provided in the course.

```
git clone https://github.com/wljungbergh/activity-recognition.git
cd activity-recognition
python3 main.py --logfile <path-to-log>
```