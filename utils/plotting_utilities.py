import matplotlib.pyplot as plt
import cv2
import numpy
class ImagePlotter:
    @staticmethod
    def get_distinct_colors(n):
        """Generate n distinct colors using HSV color space"""
        colors = []
        for i in range(n):
            hue = i / n
            colors.append(plt.cm.hsv(hue))
        return colors

    @staticmethod
    def check_collision(bbox, text_boxes, margin=20):
        """Check if a text bounding box collides with existing text boxes"""
        x1, y1, w1, h1 = bbox
        # Add margin to the collision check
        x1 -= margin
        y1 -= margin
        w1 += 2 * margin
        h1 += 2 * margin
        
        for x2, y2, w2, h2 in text_boxes:
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                return True
        return False

    @staticmethod
    def find_text_position(bbox, text_boxes, renderer, text_obj, ax, image_height, image_width):
        """Find a suitable position for text that doesn't overlap with existing text"""
        bbox_text = text_obj.get_window_extent(renderer=renderer)
        text_width = bbox_text.width
        text_height = bbox_text.height
        
        # Convert text dimensions from pixels to data coordinates
        text_width = text_width / ax.figure.dpi * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.figure.get_figwidth()
        text_height = text_height / ax.figure.dpi * (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.figure.get_figheight()
        
        # Center point of the bounding box
        center_x = bbox[0] + bbox[2]/2
        center_y = bbox[1] + bbox[3]/2
        
        # Try positions in a spiral pattern around the box
        radius = max(bbox[2], bbox[3]) * 2  # Start with twice the box size
        angle = 0
        spiral_points = 16  # Number of points to try in the spiral
        
        while radius < max(image_width, image_height):
            for i in range(spiral_points):
                angle = (i * 2 * np.pi / spiral_points)
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                # Ensure the text stays within image bounds with padding
                x = max(text_width/2, min(image_width - text_width/2, x))
                y = max(text_height/2, min(image_height - text_height/2, y))
                
                text_bbox = (x - text_width/2, y - text_height/2, text_width, text_height)
                
                if not ImagePlotter.check_collision(text_bbox, text_boxes, margin=30):
                    text_boxes.append(text_bbox)
                    return x, y, True  # True indicates we need a leader line
            
            radius += max(bbox[2], bbox[3])  # Increase radius by box size
        
        # If no position found, place it far right with increased y-spacing
        x = image_width - text_width - 10
        y = len(text_boxes) * (text_height + 20)
        text_bbox = (x, y, text_width, text_height)
        text_boxes.append(text_bbox)
        return x, y, True

    @staticmethod
    def draw_leader_line(ax, box_center, text_pos, color):
        """Draw a leader line from text to box"""
        # Create curved leader line
        connection_line = plt.patches.ConnectionPatch(
            xyA=box_center, xyB=text_pos,
            coordsA='data', coordsB='data',
            axesA=ax, axesB=ax,
            color=color, alpha=0.5,
            linestyle='-', linewidth=0.5,
            zorder=1
        )
        ax.add_artist(connection_line)

    @staticmethod
    def plot_annotations_on_image(image, annotations1, annotations2, coco_helper, output_path):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get image dimensions
        image_height, image_width = image.shape[:2]
        
        # Get renderer for text size calculations
        renderer = fig.canvas.get_renderer()
        
        # Get all unique category IDs
        category_ids = set()
        for ann in annotations1 + annotations2:
            if 'category_id' in ann:
                category_ids.add(ann['category_id'])
        
        # Create color mapping
        colors = ImagePlotter.get_distinct_colors(len(category_ids))
        color_map = {cat_id: color for cat_id, color in zip(sorted(category_ids), colors)}
        
        # Track text bounding boxes to prevent overlaps
        text_boxes = []
        
        # Track categories for legend
        categories1 = set()
        categories2 = set()
        
        # Function to plot annotations
        def plot_annotation_group(annotations, linestyle, categories_set):
            for ann in annotations:
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    category_name = coco_helper.get_category_name(ann['category_id'])
                    category_id = ann['category_id']
                    categories_set.add((category_name, category_id))
                    category_color = color_map[category_id]
                    
                    # Draw bounding box
                    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                       fill=False, edgecolor=category_color, linewidth=2,
                                       linestyle=linestyle)
                    ax.add_patch(rect)
                    
                    # Create text object to get its size
                    label_text = f"{category_name} (ID: {category_id})"
                    text_obj = ax.text(0, 0, label_text, fontsize=8)
                    
                    # Find position for text without overlap
                    box_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                    x, y, needs_leader = ImagePlotter.find_text_position(
                        bbox, text_boxes, renderer, text_obj, ax, image_height, image_width
                    )
                    
                    # Remove temporary text object
                    text_obj.remove()
                    
                    # Add text at calculated position with white background
                    text = ax.text(x, y, label_text, 
                                 color=category_color, fontsize=8,
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
                                 ha='center', va='center')
                    
                    # Draw leader line if needed
                    if needs_leader:
                        ImagePlotter.draw_leader_line(ax, box_center, (x, y), category_color)
        
        # Plot annotations for both images
        plot_annotation_group(annotations1, '--', categories1)
        plot_annotation_group(annotations2, '-', categories2)
        
        # Create legend
        legend_elements = []
        
        # Add image source indicators to legend
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', lw=2, 
                                        label='Image 1 (Transformed) - Dashed Lines'))
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', lw=2, 
                                        label='Image 2 - Solid Lines'))
        
        # Add separator in legend
        legend_elements.append(plt.Line2D([0], [0], color='none', label=''))
        
        # Add category information to legend
        legend_elements.append(plt.Line2D([0], [0], color='none', 
                                        label='Categories and Their Colors:'))
        
        # Add all unique categories with their colors
        all_categories = sorted(categories1.union(categories2))
        for cat_name, cat_id in all_categories:
            legend_elements.append(plt.Line2D([0], [0], color=color_map[cat_id], lw=2,
                                            label=f'{cat_name} (ID: {cat_id})'))
        
        # Add section for categories present in each image
        legend_elements.append(plt.Line2D([0], [0], color='none', label=''))
        legend_elements.append(plt.Line2D([0], [0], color='none', 
                                        label='Categories present in:'))
        legend_elements.append(plt.Line2D([0], [0], color='none', 
                                        label='Image 1: ' + 
                                        ', '.join(f'{cat[0]} ({cat[1]})' 
                                                for cat in sorted(categories1))))
        legend_elements.append(plt.Line2D([0], [0], color='none', 
                                        label='Image 2: ' + 
                                        ', '.join(f'{cat[0]} ({cat[1]})' 
                                                for cat in sorted(categories2))))
        
        # Add legend to plot
        ax.legend(handles=legend_elements, 
                 loc='center left',
                 bbox_to_anchor=(1, 0.5),
                 fontsize=10,
                 frameon=True,
                 facecolor='white',
                 edgecolor='black',
                 title='Legend')
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save with extra space for legend
        plt.savefig(output_path, 
                   bbox_inches='tight',
                   pad_inches=0.5,
                   dpi=300)
        plt.close()