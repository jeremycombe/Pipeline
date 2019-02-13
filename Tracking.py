from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
from matplotlib import pyplot as plt


# Class Setup

class OpenVinoObjectDetectionModel(object):

    def __init__(self, **kwargs):
        """
        Builds an OpenVINO model.

        Keyword arguments (in order):
        model_path: Path to an .xml file with a trained model.
        cpu_extension: MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.
        plugin_dir: Path to a plugin folder
        device: Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. CPU by default.
        labels_path: Labels mapping file (format .labels)
        prob_threshold: Probability threshold for detections filtering. Float between 0.0 and 1.0.
        """

        self.__dict__.update(kwargs)
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        self.generate(**kwargs)
        log.info("Model initialized and loaded.")

    def generate(self, model_path, cpu_extension=None, plugin_dir=None, device="CPU",
                 labels_path=None, prob_threshold=0.5):

        self.model_xml = model_path
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"

        # Plugin initialization for specified device and load extensions library if specified
        log.info("Initializing plugin for {} device...".format(device))
        self.plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)

        # Read IR
        log.info("Reading IR...")
        self.net = IENetwork.from_ir(model=self.model_xml, weights=self.model_bin)

        if "CPU" in self.plugin.device:
            supported_layers = self.plugin.get_supported_layers(self.net)
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(self.plugin.device, ', '.join(not_supported_layers)))
                log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                raise ValueError(
                    "Some layers are not supported by the plugin for the specified device {}".format(device))

        assert len(self.net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(self.net.outputs) == 1, "Sample supports only single output topologies"
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        log.info("Loading IR to the plugin...")
        self.exec_net = self.plugin.load(network=self.net, num_requests=2)
        self.cur_request_id = 0
        self.next_request_id = 1

        self.n, self.c, self.h, self.w = self.net.inputs[self.input_blob].shape
        del self.net

    def detect_objects(self, image, resolution):
        """
        Runs inference on the supplied image.

        Keyword arguments:
        image: Image to be inferenced on
        resolution: Tuple of (width, height) of the image
        """
        is_async_mode = False

        image = cv2.resize(image, (self.w, self.h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image.reshape((self.n, self.c, self.h, self.w))

        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion

        if is_async_mode:
            self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: image})
        else:
            self.exec_net.start_async(request_id=self.cur_request_id, inputs={self.input_blob: image})

        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:

            # Parse detection results of the current request
            res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
            bboxes = []

            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                confidence = obj[2]
                if confidence > self.prob_threshold:
                    xmin = int(obj[3] * resolution[0])
                    ymin = int(obj[4] * resolution[1])
                    xmax = int(obj[5] * resolution[0])
                    ymax = int(obj[6] * resolution[1])
                    class_id = int(obj[1])
                    bboxes.append((class_id, confidence, xmin, ymin, xmax, ymax))

        self.next_request_id, self.cur_request_id = self.cur_request_id, self.next_request_id
        return image, bboxes

    def detect_objects_partition(self, subimages, partition):
        """
        Runs inference on the supplied subimages. Returns bboxes with original image coordinates

        Keyword arguments:
        subimages: List of images to be inferenced on
        partition: Dict with keys ("xmins", "xmaxs", "ymins", "ymaxs"), each of which contains
                   a list of coordinates corresponding to the subimages
        """
        bboxes = []

        for i, image in enumerate(subimages):
            image = cv2.resize(image, (self.w, self.h))
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            image = image.reshape((self.n, self.c, self.h, self.w))

            self.exec_net.start_async(request_id=0, inputs={self.input_blob: image})

            if self.exec_net.requests[0].wait(-1) == 0:

                part_height = partition["ymaxs"][i] - partition["ymins"][i]
                part_width = partition["xmaxs"][i] - partition["xmins"][i]

                # Parse detection results of the current request
                res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]

                for obj in res[0][0]:
                    # Draw only objects when probability more than specified threshold
                    confidence = obj[2]
                    if confidence > self.prob_threshold:
                        xmin = int(obj[3] * part_width) + partition["xmins"][i]
                        ymin = int(obj[4] * part_height) + partition["ymins"][i]
                        xmax = int(obj[5] * part_width) + partition["xmins"][i]
                        ymax = int(obj[6] * part_height) + partition["ymins"][i]
                        class_id = int(obj[1])
                        bboxes.append((class_id, confidence, xmin, ymin, xmax, ymax))

        return bboxes


# Detection and tracking model setup

class DetectionAndTrackingController(object):

    def __init__(self, model, classes, tracker_type, drawer,
                 resize_resolution=None, tracking_resolution=(640, 360)):

        self.timer = Timer()
        self.timer.start_task("Setup")

        self.model = model
        self.classes = classes
        self.tracker_type = tracker_type
        self.drawer = drawer

        if resize_resolution is not None:
            self.resolution = resize_resolution
            self.resize_resolution = resize_resolution
        else:
            self.resolution = None

        self.tracking_resolution = tracking_resolution

        self.model_name = "OpenVINO Vehicle Detection"

    def execute(self, vid_source, output_path=None, sampling=12, lane_edges=[0.00, 1.00],
                lane_directions=["down"], finish_line=0.7, max_line=0.4, image_partitioning=True):
        """
        Starts a session of detection and tracking on the provided video using provided data.

        Keyword arguments:
        vid_source:         String. Full path to video or video stream
        output_path:        String. Full path to file where the output will be saved.
                            If not provided, no video will be saved.
        sampling:           Int. Frequency of vehicle detection step. Defaults to 12.
        lane_edges:         List of floats between 0 and 1, must be in increasing order.
                            Denotes the edges between lanes (from left to right)
                            as a percentage of the frame width. Must have (number_of_lanes + 1) items.
        lane_directions:    List of strings. Denotes the directions of the lanes. Each item must
                            be "up" or "down". Must have (number_of_lanes) items.
        finish_line:        Float between 0 and 1. Position of finish line as percentage of video height.
        max_line:           Float between 0 and 1. Position of max line as percentage of video height.
        image_partitioning: Boolean. Denotes whether to use image partitioning for detection.
        """

        self._initialize_run(vid_source, output_path, sampling,
                             lane_edges, lane_directions, finish_line, max_line)

        # Initialize time variables to measure FPS
        time_start = time.time()
        time_this = time.time()
        time_prev = time.time()

        # ret will become false when all frames in the input video has been read
        ret = True

        while (ret):

            self.timer.start_task("Video/image processing")

            # Get next frame from video
            ret, self.current_image = self.cap.read()

            # If there are no more frames, exit the loop
            if ret == False:
                break

            # Resize if needed
            if self.resize_resolution is not None:
                self.current_image = cv2.resize(self.current_image, self.resolution)

            self.current_image_tracking = cv2.resize(self.current_image, self.tracking_resolution)

            self.counters["frames"] += 1

            self.timer.start_task("Managing trackers")

            # Update trackers and boxes
            self._update_trackers()

            # If this is the Nth frame, perform classification. Otherwise, follow the tracking
            if self.counters["frames"] % self.sampling == 0:

                self.timer.start_task("Detection")
                if image_partitioning:
                    partition, subimages = self._partition_image()
                    bboxes_detected = self.model.detect_objects_partition(subimages, partition)
                else:
                    _, bboxes_detected = self.model.detect_objects(current_image, self.resolution)

                self.timer.start_task("Processing detection")
                self._process_detections(bboxes_detected)

            self.timer.start_task("Drawing on image")

            # Calculate FPS
            time_this = time.time()
            fps = 1 / (time_this - time_prev + 0.00001)
            time_prev = time.time()

            # Draw overlay
            self.current_image = self.drawer.draw_overlay(self.current_image, self.counters,
                                                          self.lane_edges, self.finish_line,
                                                          self.max_line, fps)

            # Draw bboxes on the image
            self.current_image = self.drawer.draw_bboxes_on_image(self.current_image,
                                                                  self.trackers, self.classes)

            # Clean up finished vehicles
            self.trackers = [tracker for n, tracker in enumerate(self.trackers) if tracker["finished"] == False]

            self.timer.start_task("Video display")

            # Write frame to output if applicable
            if self.output_path is not None:
                self.video_output.write(self.current_image)

            # Show output in window
            cv2.imshow(self.model_name, self.current_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        self._finalize_run()

    def _initialize_run(self, vid_source, output_path, sampling,
                        lane_edges, lane_directions, finish_line, max_line):
        self.sampling = sampling
        self.output_path = output_path
        self.lane_edges = lane_edges
        self.lane_directions = lane_directions
        self.num_lanes = len(self.lane_directions)

        self.trackers = []
        self.counters = {
            "n_vehicles": 0,
            "lost_trackers": 0,
            "frames": 0,
        }

        for i in range(self.num_lanes):
            self.counters["lane{}".format(i)] = 0

        self.cap = cv2.VideoCapture(vid_source)

        # If an output path is specified, create a video output with the same attributes
        # as the input video
        framerate = self.cap.get(cv2.CAP_PROP_FPS)

        if self.resolution is None:
            self.resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.output_path is not None:
            codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # fourcc stands for four character code
            self.video_output = cv2.VideoWriter(self.output_path, codec, self.framerate, self.resolution)
        self.timer.stop()

        self.lane_edges = [int(self.resolution[0] * x) for x in lane_edges]
        self.finish_line = int(self.resolution[1] * finish_line)
        self.max_line = int(self.resolution[1] * max_line)

    def _finalize_run(self):
        cv2.destroyAllWindows()
        self.cap.release()
        if self.output_path is not None:
            self.video_output.release()

        self.timer.stop()
        timers = self.timer.get_timers()

        print("\n======== Timer summary ========")
        for key in timers:
            print("{}:{} {:0.4f} seconds \t({:0.4f} s per frame)".format(key, " " * (25 - len(key)), timers[key],
                                                                         timers[key] / self.counters["frames"]))
        print("\nDetection:{} {:0.4f} seconds per detection every {} frames".format(" " * (25 - 9),
                                                                                    timers["Detection"] / (
                                                                                                self.counters[
                                                                                                    "frames"] // self.sampling),
                                                                                    self.sampling))

    def _update_trackers(self):
        lane_counts = [self.counters["lane" + str(i)] for i in range(self.num_lanes)]

        for n, tracker_data in enumerate(self.trackers):
            v_type = "vehicle_{}".format(self.classes[tracker_data["label"]])
            success, bbox = tracker_data["tracker"].update(self.current_image_tracking)

            if not success:
                self.counters["lost_trackers"] += 1
                tracker_data["lost"] = True
                continue

            # Extract bbox coordinates and transform to original resolution
            xmin = int(bbox[0] * self.resolution[0] / self.tracking_resolution[0])
            ymin = int(bbox[1] * self.resolution[1] / self.tracking_resolution[1])
            xmax = int((bbox[0] + bbox[2]) * self.resolution[0] / self.tracking_resolution[0])
            ymax = int((bbox[1] + bbox[3]) * self.resolution[1] / self.tracking_resolution[1])
            xmid = int(round((xmin + xmax) / 2))
            ymid = int(round((ymin + ymax) / 2))

            # If the car has crossed the finish line, add 1 to count and mark tracker as finished.
            for i in range(self.num_lanes):
                if xmid >= self.lane_edges[i] and xmid < self.lane_edges[i + 1]:
                    if self.lane_directions[i] == "down" and ymid >= self.finish_line:
                        lane_counts[i] += 1
                        self.counters[v_type] = self.counters.get(v_type, 0) + 1
                        tracker_data["finished"] = True
                    elif self.lane_directions[i] == "up" and ymid <= self.finish_line:
                        lane_counts[i] += 1
                        self.counters[v_type] = self.counters.get(v_type, 0) + 1
                        tracker_data["finished"] = True

            tracker_data["bbox"] = (xmin, ymin, xmax, ymax)  # Update bbox for this tracker

        # Clean up lost vehicles
        self.trackers = [tracker for n, tracker in enumerate(self.trackers) if tracker["lost"] == False]

        # Write new lane counts to the counters dict
        for i in range(self.num_lanes):
            self.counters["lane{}".format(i)] = lane_counts[i]

    def _in_range(self, bbox):
        xmin = bbox[2]
        ymin = bbox[3]
        xmax = bbox[4]
        ymax = bbox[5]
        xmid = int(round((xmin + xmax) / 2))
        ymid = int(round((ymin + ymax) / 2))

        # Disregard boxes that are obviously too big
        if (xmax - xmin) + (ymax - ymin) > self.resolution[0] * 0.6:
            return False

        # If the car is in a lane going upwards (away from the camera), the valid range is below the finish line
        # If the car is in a lane going downwards, the valid range is above the finish line and below the start line
        for i in range(len(self.lane_directions)):
            if xmid >= self.lane_edges[i] and xmid < self.lane_edges[i + 1]:
                if self.lane_directions[i] == "down" and (ymid > self.finish_line or ymid < self.max_line):
                    return False
                elif self.lane_directions[i] == "up" and ymid < self.finish_line:
                    return False
                else:
                    return True
        return False

    def _add_new_tracker(self, bbox, car_id):
        label = bbox[0]
        confidence = bbox[1]
        # Scale coordinates to the tracking resolution
        xmin = int(bbox[2] * self.tracking_resolution[0] / self.resolution[0])
        xmax = int(bbox[4] * self.tracking_resolution[0] / self.resolution[0])
        ymin = int(bbox[3] * self.tracking_resolution[1] / self.resolution[1])
        ymax = int(bbox[5] * self.tracking_resolution[1] / self.resolution[1])
        xmid = int(round((xmin + xmax) / 2))
        ymid = int(round((ymin + ymax) / 2))

        # init tracker
        tracker = self.tracker_type()
        success = tracker.init(self.current_image_tracking, (xmin, ymin, xmax - xmin, ymax - ymin))
        if success:
            self.trackers.append({"tracker": tracker,
                                  "car_id": str(car_id),
                                  "label": label,
                                  "confidence": confidence,
                                  "bbox": (xmin, ymin, xmax, ymax)})

    def _process_detections(self, boxes_detected):
        max_class_id = len(self.classes) - 1
        # Iterate over all detected boxes and check if they are already tracked or not
        new_boxes = []
        for bbox_d in boxes_detected:
            # Reminder: bbox_d is of format [label, confidence, xmin, ymin, xmax, ymax]
            # Thus bbox_d[2:] returns only the coordinates of the bounding box
            if self._in_range(bbox_d) and int(bbox_d[0]) <= max_class_id:
                ymin, ymax, xmin, xmax = bbox_d[3], bbox_d[5], bbox_d[2], bbox_d[4]

                # Iterate over all tracked boxes (using a generator expression)
                # to check if they match with this detection.
                # This is done by checking if the IOU is greater than 0.4.
                tracker_data = next((tracker_data for tracker_data in self.trackers
                                     if self._get_iou(bbox_d[2:], tracker_data["bbox"]) >= 0.4), None)

                if tracker_data is None:
                    # Did not find an overlapping tracker; initialize a new tracker
                    self.counters["n_vehicles"] += 1
                    tracker = self._get_new_tracker(bbox_d[2:])
                    self.trackers.append({"tracker": tracker,
                                          "vehicle_id": str(self.counters["n_vehicles"]),
                                          "label": bbox_d[0],
                                          "confidence": bbox_d[1],
                                          "finished": False,
                                          "lost": False,
                                          "bbox": (xmin, ymin, xmax, ymax)})

                else:
                    # Found an existing overlapping tracker; reinitialize the tracker with the new box
                    tracker_data["tracker"] = self._get_new_tracker(bbox_d[2:])
                    tracker_data["label"] = bbox_d[0]
                    tracker_data["confidence"] = bbox_d[1]
                    tracker_data["bbox"] = (xmin, ymin, xmax, ymax)

    def _get_new_tracker(self, bbox):
        """
        Returns a new tracker, or raises a RuntimeError if initialisation is not successful.

        Keyword arguments:
        bbox:   List-like of form [xmin, ymin, xmax, ymax]
        """
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        tracker = self.tracker_type()
        # The tracker init() function takes (image, xmin, ymin, width, height) as input
        success = tracker.init(self.current_image_tracking,
                               (int(xmin * self.tracking_resolution[0] / self.resolution[0]),
                                int(ymin * self.tracking_resolution[1] / self.resolution[1]),
                                int((xmax - xmin) * self.tracking_resolution[0] / self.resolution[0]),
                                int((ymax - ymin) * self.tracking_resolution[1] / self.resolution[1])))
        if success:
            return tracker
        else:
            raise RuntimeError("Tracker not initialised!")

    def _get_iou(self, box1, box2):
        """Returns the intersection over union (IoU) between box1 and box2

        Arguments:
        box1 -- first box, list object with coordinates (x1, y1, x2, y2)
        box2 -- second box, list object with coordinates (x1, y1, x2, y2)
        """

        # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = (xi2 - xi1) * (yi2 - yi1)

        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        # compute the IoU
        iou = inter_area / union_area

        return iou

    def _partition_image(self):
        # Define groups of lanes - lanes next to each other that go in the same direction
        groups = []
        current_group = []
        for n, lane_direction in enumerate(self.lane_directions):
            # Add this lane to group if the group is empty
            if len(current_group) == 0:
                current_group.append(n)
            # Add this lane to group if the group has the same direction
            elif self.lane_directions[current_group[0]] == lane_direction:
                current_group.append(n)
            # Otherwise, finalise group and start a new group
            else:
                groups.append(current_group)
                current_group = [n]
        groups.append(current_group)

        # Calculate coordinates of sub-images
        partition = {"xmins": [], "xmaxs": [], "ymins": [], "ymaxs": []}

        for group in groups:
            # Use the left edge of the first lane in the group and the right edge of the last lane in the group
            partition["xmins"].append(self.lane_edges[group[0]])
            partition["xmaxs"].append(self.lane_edges[group[-1] + 1])

            # If lane goes downwards we want from the top of ROI to the finish line, with some padding
            if self.lane_directions[group[0]] == "down":
                partition["ymins"].append(max(self.max_line - 40, 0))
                partition["ymaxs"].append(min(self.finish_line + 40, self.resolution[1]))
            # If lane goes upwards we want from the finish line to the bottom of the image, with some padding
            elif self.lane_directions[group[0]] == "up":
                partition["ymins"].append(max(self.finish_line - 40, 0))
                partition["ymaxs"].append(self.resolution[1])

        # Then do the partitioning of the image. Use the image[ymin:ymax, xmin:xmax] notation
        subimages = []
        for i in range(len(groups)):
            subimages.append(self.current_image[partition["ymins"][i]:partition["ymaxs"][i],
                             partition["xmins"][i]:partition["xmaxs"][i]])

        return partition, subimages


# Drawer Object

class OverlayDrawer(object):

    def __init__(self, num_classes):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_small = 0.5
        self.font_scale_large = 1.0
        self.thickness_small = 1
        self.thickness_medium = 2
        self.thickness_large = 4
        self.padding = 4
        self.colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # Some standard colors to use
        self.c_finish_line = (255, 255, 224)  # Light cyan
        self.c_max_line = (128, 128, 128)  # Gray
        self.c_counter = (0, 0, 230)  # Red
        self.c_finished = (80, 220, 60)  # Green
        self.c_fps = (80, 220, 60)  # Green
        self.c_white = (255, 255, 255)  # White

    def draw_bboxes_on_image(self, image, trackers, classes):
        for n, tracker_data in enumerate(trackers):
            bbox = tracker_data["bbox"]
            classification = int(tracker_data["label"])
            confidence = tracker_data["confidence"]
            color = self.colors[classification]

            if tracker_data["finished"]:
                color = self.c_finished

            label = "{} {:.2f}".format(classes[classification], confidence)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]

            # Draw main box
            image = cv2.rectangle(image, (left, top), (right, bottom), color, 2)

            # Draw label box above the top left corner of the main box
            label_size = cv2.getTextSize(label, self.font, self.font_scale_small, self.thickness_small)
            label_width = int(label_size[0][0])
            label_height = int(label_size[0][1])

            # If there is space above the box, draw the label there; else draw it below
            if top - label_height > 0:
                label_top = top - label_height - self.padding
            else:
                label_top = bottom

            label_bottom = label_top + label_height + self.padding
            label_left = left
            label_right = left + label_width + self.padding
            image = cv2.rectangle(image, (label_left, label_top), (label_right, label_bottom), color, -1)
            image = cv2.putText(image, label,
                                (label_left + int(self.padding * 0.5), label_bottom - int(self.padding * 0.5)),
                                self.font, self.font_scale_small, self.c_white, self.thickness_small)

            # Draw vehicle ID number
            vehicle_id = str(tracker_data["vehicle_id"])
            id_size, _ = cv2.getTextSize(vehicle_id, self.font, self.font_scale_large, self.thickness_large)
            pos = (left + (right - left) // 2 - id_size[0] // 2,
                   top + (bottom - top) // 2 + id_size[1] // 2)
            cv2.putText(image, vehicle_id, pos, self.font, self.font_scale_large, color, self.thickness_large)

        return image

    def draw_overlay(self, image, counters, lane_edges, finish_line, max_line, fps):
        num_lanes = len(lane_edges) - 1
        lane_counts = [counters["lane{}".format(i)] for i in range(num_lanes)]
        resolution = (image.shape[1], image.shape[0])

        # Draw start line, if > 0
        if max_line > 0:
            cv2.line(image, (0, max_line), (resolution[0], max_line), self.c_max_line, self.thickness_medium)

        # Draw finish line with lane hash marks
        cv2.line(image, (0, finish_line), (resolution[0], finish_line), self.c_finish_line, self.thickness_large)
        for edge in lane_edges:
            cv2.line(image, (edge, finish_line - 20), (edge, finish_line + 20), self.c_finish_line,
                     self.thickness_large)

        # Add lane counter
        for i in range(num_lanes):
            cv2.putText(image, str(lane_counts[i]), (lane_edges[i] + 10, finish_line + 50),
                        self.font, self.font_scale_large, self.c_finish_line, self.thickness_medium)

        # Add vehicle type counter
        v_type_count = 0
        for key in counters:
            if key[0:7] == "vehicle":
                cv2.putText(image, "{}: {}".format(key[8:], counters[key]), (int(resolution[0] * 0.85),
                                                                             40 + 30 * v_type_count), self.font,
                            self.font_scale_large,
                            self.c_counter, self.thickness_medium)
                v_type_count += 1

        # Add FPS
        cv2.putText(image, "FPS: {:.2f}".format(fps), (3, 25), self.font, self.font_scale_large,
                    self.c_fps, self.thickness_medium)

        # Draw the running total of cars in the image in the upper-left corner
        cv2.putText(image, 'Cars detected: ' + str(counters["n_vehicles"]), (3, 60),
                    self.font, self.font_scale_large, self.c_counter, self.thickness_medium)

        # Add note with count of trackers lost
        cv2.putText(image, 'Cars lost: ' + str(counters["lost_trackers"]), (3, 85),
                    self.font, self.font_scale_small, self.c_counter, self.thickness_small)

        return image


# Timer object


import time


class Timer(object):
    # Helper class to time task. Every tile start(task) is called, the previous task is stopped and recorded

    def __init__(self):
        self._timers = {}
        self._time_now = time.time()
        self._time_prev = time.time()
        self._curr_task = None

    def start_task(self, task):
        self._time_now = time.time()
        time_passed = self._time_now - self._time_prev
        if self._curr_task is not None:
            self._timers[self._curr_task] = self._timers.get(self._curr_task, 0.0) + time_passed
        self._curr_task = task
        self._time_prev = time.time()

    def stop(self):
        self._time_now = time.time()
        time_passed = self._time_now - self._time_prev
        if self._curr_task is not None:
            self._timers[self._curr_task] = self._timers.get(self._curr_task, 0.0) + time_passed
        self._curr_task = None

    def get_timers(self):
        return self._timers


# Fixed parametres

### Fixed parameters. Normally do not need to be changed unless adding new models ###

# Should be validated when running on a new installation of OpenVINO
model_dir = "/home/openvino/openvino_models"

# Class names
classes_mscoco = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
           'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
           'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
           'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
           'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
           'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']
classes_intel = ["bg", "car", "motorcycle", "pedestrian"]

# Device management
data_type = {"CPU": "FP32", "GPU": "FP16", "MYRIAD": "FP16"}

models = ["Intel vehicle detection",
          "Intel crossroad detection",
          "TF SSD MobileNet v2",
          "TF SSD MobileNet FPN",
          "TF SSDlite MobileNet v2"]
model_paths = {"Intel vehicle detection": ("vehicle-detection-adas-0002","vehicle-detection-adas-0002.xml"),
          "Intel crossroad detection": ("person-vehicle-bike-detection-crossroad-0078","person-vehicle-bike-detection-crossroad-0078.xml"),
          "TF SSD MobileNet v2": ("tensorflow_ssd_mobilenet_v2_coco","ssd_mobilenet_v2_coco.xml"),
          "TF SSD MobileNet FPN": ("ssd_mobilenet_v1_fpn_640x640","ssd_mobilenet_v1_fpn_640x640.xml"),
          "TF SSDlite MobileNet v2": ("ssdlite_mobilenet_v2", "ssdlite_mobilenet_v2.xml")
           }
classes = {"Intel vehicle detection": classes_intel,
          "Intel crossroad detection": classes_intel,
          "TF SSD MobileNet v2": classes_mscoco,
          "TF SSD MobileNet FPN": classes_mscoco,
          "TF SSDlite MobileNet v2": classes_mscoco
           }

# Available trackers

## IMPORTANT: "cv2_contrib" is just a hack to be able to use tracking,
## which is not available in the OpenVINO-accelerated OpenCV.
## Ideally opencv-contrib should be built into this.
import cv2_contrib

OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2_contrib.TrackerCSRT_create,
        "kcf": cv2_contrib.TrackerKCF_create,
        "boosting": cv2_contrib.TrackerBoosting_create,
        "mil": cv2_contrib.TrackerMIL_create,
        "tld": cv2_contrib.TrackerTLD_create,
        "medianflow": cv2_contrib.TrackerMedianFlow_create,
        "mosse": cv2_contrib.TrackerMOSSE_create,
        "goturn": cv2_contrib.TrackerGOTURN_create
    }