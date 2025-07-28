#!/usr/bin/python
import os
import rospy
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from graspnet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from graspnet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from graspnet.networks.graspnet_model import GraspNet
from sklearn.cluster import DBSCAN, HDBSCAN
from graspnet.msg import GraspMessage

class GraspInference:
    def __init__(self):
        # ROS setup
        rospy.init_node("grasp_inference", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/nakit_vision/color/image_raw", Image, self.image_callback)
        
        # Parameters
        self.visualize = rospy.get_param('~visualize', False)  # Default: False
        self.heatmap_threshold = rospy.get_param('~heatmap_threshold', 0.3)
        
        # Output publishers
        self.grasp_pub = rospy.Publisher('/grasp_detections', GraspMessage, queue_size=1)
        self.class_pub = rospy.Publisher('/object_classes', Float32MultiArray, queue_size=1)
        self.overlay_pub = rospy.Publisher('/grasp_inference/overlay_seg', Image, queue_size=1)
        self.heatmap_pub = rospy.Publisher('/grasp_inference/overlay_heatmap', Image, queue_size=1)
        
        # Neural network setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_seg_classes = 26
        self.input_channels = 3 + self.num_seg_classes
        self.img_size = 480

        # Load TransUNet
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = self.num_seg_classes
        config_vit.n_skip = 3
        patch_size = config_vit.patches["size"]
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        config_vit.patches.grid = (self.img_size // patch_size, self.img_size // patch_size)

        self.transunet = ViT_seg(config_vit, img_size=self.img_size, num_classes=self.num_seg_classes).to(self.device)
        print(f"DEBUG: dir={os.getcwd()}")
        state_dict = torch.load("/root/graspnet_ws/src/graspnet/src/graspnet/TU_mixed480/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs4_480/epoch_149.pth")
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('grasp_head')}
        self.transunet.load_state_dict(state_dict, strict=False)
        self.transunet.eval()

        # Load GraspNet
        self.graspnet = GraspNet(input_channels=self.input_channels).to(self.device)
        self.graspnet.load_state_dict(torch.load("/root/graspnet_ws/src/graspnet/src/graspnet/graspnet.pth"))
        self.graspnet.eval()

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_image_resized = cv_image[:480, 70:550]
            image_tensor = torch.tensor(cv_image_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

            # Perform inference
            seg_probs, heatmap_pred, orientation_pred, confidence_pred = self.run_inference(image_tensor)
            
            # Process and publish results
            present_classes = self.process_results(image_tensor, seg_probs, heatmap_pred, orientation_pred, confidence_pred)
            
            overlay = self.create_segmentation_overlay(image_tensor, seg_probs)
            hm_overlay = self.create_heatmap_overlay(image_tensor, heatmap_pred, orientation_pred, present_classes)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            heatmap_msg = self.bridge.cv2_to_imgmsg(hm_overlay, encoding="bgr8")
            self.overlay_pub.publish(overlay_msg)
            self.heatmap_pub.publish(heatmap_msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    @torch.no_grad()
    def run_inference(self, image_tensor):
        # Get segmentation
        seg_logits = self.transunet(image_tensor)
        seg_probs = torch.softmax(seg_logits, dim=1)
        
        # Get predicted class (channel with max probability)
        pred_class = torch.argmax(seg_probs, dim=1).cpu().numpy()[0]

        # Forward GraspNet
        x = torch.cat([image_tensor, seg_probs], dim=1)
        heatmap_pred, orientation_pred, confidence_pred = self.graspnet(x)
        
        return seg_probs, heatmap_pred, orientation_pred, confidence_pred

    def process_results(self, image_tensor, seg_probs, heatmap_pred, orientation_pred, confidence_pred):
        seg_probs_np = seg_probs[0].cpu().numpy()  # [C, H, W]
        heatmap_np = heatmap_pred[0, 0].cpu().numpy()
        orientation_np = orientation_pred[0].cpu().numpy()

        present_classes = []
        grasp_data = []

        for class_id in range(self.num_seg_classes):
            class_prob_map = seg_probs_np[class_id]
            presence_prob = float(class_prob_map.mean())
            
            if presence_prob > 0.03:  # filter out low-confidence classes
                present_classes.append((class_id, presence_prob))

                # Mask the heatmap with the segmentation class
                mask = class_prob_map > 0.3  # pixel-level thresholding
                heatmap_masked = heatmap_np * mask
                y, x = np.where(heatmap_masked > self.heatmap_threshold)

                if len(x) > 0:
                    # Prepare data for clustering
                    points = np.column_stack((x, y))
                    clustering = DBSCAN(eps=10, min_samples=40).fit(points)

                    for cluster_id in set(clustering.labels_):
                        if cluster_id == -1:  # Ignore noise points
                            continue

                        # Extract points in the current cluster
                        cluster_points = points[clustering.labels_ == cluster_id]
                        cluster_center = cluster_points.mean(axis=0).astype(int)

                        # Get confidence at the cluster center
                        cx, cy = cluster_center
                        confidence = heatmap_np[cy, cx]

                        # Get orientations as the average orientation in the cluster
                        orientations = orientation_np[:, y, x]
                        u = orientations[0, clustering.labels_ == cluster_id].mean()
                        v = orientations[1, clustering.labels_ == cluster_id].mean()    

                        grasp_data.append([class_id, presence_prob, cx, cy, u, v, confidence])
        
                        # Create and publish GraspMessage
                        grasp_msg = GraspMessage()
                        grasp_msg.header.stamp = rospy.Time.now()
                        grasp_msg.class_id = int(class_id)  # Ensure it's an integer
                        grasp_msg.presence_prob = float(presence_prob)  # Ensure it's a float
                        grasp_msg.cx = int(cx)  # Ensure it's an integer
                        grasp_msg.cy = int(cy)  # Ensure it's an integer
                        grasp_msg.u = float(u)  # Ensure it's a float
                        grasp_msg.v = float(v)  # Ensure it's a float
                        grasp_msg.confidence = float(confidence)  # Ensure it's a float
        
                        self.grasp_pub.publish(grasp_msg)

        rospy.loginfo(f"Detected classes: {present_classes}")
        rospy.loginfo(f"Published {len(grasp_data)} grasp points across {len(present_classes)} classes")

        return present_classes

    def create_segmentation_overlay(self, image_tensor, seg_probs):
        image_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pred_class = torch.argmax(seg_probs, dim=1).cpu().numpy()[0]

        overlay = image_np.copy()

        # Define 26 distinct colors
        colors = [
            (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255),
            (0,255,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0),
            (128,0,128), (0,128,128), (192,192,192), (128,64,0), (0,128,64),
            (64,0,128), (255,128,0), (0,255,128), (128,0,255), (255,0,128),
            (128,255,0), (0,128,255), (200,100,50), (50,200,100), (100,50,200),
            (150,150,0)
        ]

        for class_id in range(1, self.num_seg_classes):  # skip background
            mask = (pred_class == class_id).astype(np.uint8)
            if np.any(mask):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, colors[class_id % len(colors)], 2)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cv2.putText(overlay, str(class_id), (cX, cY),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id % len(colors)], 2)
        return overlay

        
    def create_heatmap_overlay(self, image_tensor, heatmap_pred, orientation_pred, present_classes=None):
        image_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        heatmap_np = heatmap_pred[0, 0].cpu().numpy()
        orientation_np = orientation_pred[0].cpu().numpy()

        # Normalize heatmap to [0, 255]
        heatmap_normalized = cv2.normalize(heatmap_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # Threshold the heatmap to find high-confidence regions
        y, x = np.where(heatmap_np > self.heatmap_threshold)
        if len(x) == 0:
            rospy.logwarn("No points found in heatmap for overlay")
            return cv2.addWeighted(image_np, 0.7, heatmap_colored, 0.3, 0)

        # Prepare data for clustering
        points = np.column_stack((x, y))

        # Perform HDBSCAN clustering
        clustering = HDBSCAN(min_cluster_size=20, min_samples=20).fit(points)

        # Overlay heatmap on the original image
        overlay = cv2.addWeighted(image_np, 0.7, heatmap_colored, 0.3, 0)

        # Track the highest confidence cluster for each class
        highest_confidence_clusters = {}

        for class_id, _ in present_classes:
            highest_confidence_clusters[class_id] = None  # Initialize with None

        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Ignore noise points
                continue

            # Extract points in the current cluster
            cluster_points = points[clustering.labels_ == cluster_id]
            cluster_center = cluster_points.mean(axis=0).astype(int)

            # Calculate the average confidence for the cluster
            cluster_confidence = heatmap_np[cluster_points[:, 1], cluster_points[:, 0]].mean()

            # Assign the cluster to the corresponding class
            for class_id, _ in present_classes:
                if highest_confidence_clusters[class_id] is None or cluster_confidence > highest_confidence_clusters[class_id][1]:
                    highest_confidence_clusters[class_id] = (cluster_id, cluster_confidence)

        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Ignore noise points
                continue

            # Extract points in the current cluster
            cluster_points = points[clustering.labels_ == cluster_id]
            cluster_center = cluster_points.mean(axis=0).astype(int)

            # Draw contour around the cluster
            cluster_mask = np.zeros_like(heatmap_np, dtype=np.uint8)
            cluster_mask[cluster_points[:, 1], cluster_points[:, 0]] = 255
            contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Determine the color for the contour
            color = (0, 255, 0)  # Default: green
            for class_id, (highest_cluster_id, _) in highest_confidence_clusters.items():
                if cluster_id == highest_cluster_id:
                    color = (0, 0, 255)  # Red for the highest confidence cluster of this class

            cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw orientation arrow at the cluster center
            cx, cy = cluster_center
            u = orientation_np[0, cy, cx]
            v = orientation_np[1, cy, cx]
            arrow_end = (int(cx + u * 20), int(cy + v * 20))  # Scale arrow length
            cv2.arrowedLine(overlay, (cx, cy), arrow_end, (255, 0, 0), 2, tipLength=0.3)

        return overlay

if __name__ == "__main__":
    try:
        GraspInference()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass