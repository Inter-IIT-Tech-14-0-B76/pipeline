import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from prompt import Segmentation


# ------------------------------------------------------------
# Dim background using mask
# ------------------------------------------------------------
def visualize_with_mask(image_np, mask):
    img = image_np.copy()

    dimmed = img // 3
    output = img.copy()
    output[mask == 0] = dimmed[mask == 0]

    return output


# ------------------------------------------------------------
# Show coordinate grid for user reference
# ------------------------------------------------------------
def show_image_with_coordinates(image_np):
    plt.figure(figsize=(10, 8))
    plt.imshow(image_np)
    plt.title("Image (with pixel coordinates)")
    
    # keep default matplotlib orientation (bottom-left origin for axis labels)
    plt.axis("on")

    h, w = image_np.shape[:2]
    plt.xticks(np.arange(0, w, max(1, w // 20)))
    plt.yticks(np.arange(0, h, max(1, h // 20)))

    plt.show(block=False)
    plt.pause(0.001)

# ------------------------------------------------------------
# Get a point from clicking on the matplotlib window
# ------------------------------------------------------------
clicked_point = None

def get_point_from_click(image_np):
    global clicked_point
    clicked_point = None

    def onclick(event):
        global clicked_point
        if event.xdata is None or event.ydata is None:
            return
        clicked_point = (int(event.xdata), int(event.ydata))
        print("Clicked:", clicked_point)
        plt.close(fig) 

    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.set_title("Click anywhere to select a point")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Use non-blocking wait that closes immediately on click
    plt.show(block=False)
    plt.waitforbuttonpress(0)   # <-- waits until ANY click

    fig.canvas.mpl_disconnect(cid)
    return clicked_point

# ============================================================
# CLICK HELPERS
# ============================================================

# ------------------------------
# 1. POINT CLICK
# ------------------------------
def get_point_from_click(image_np):
    result = {"pt": None}

    def onclick(event):
        if event.xdata and event.ydata:
            result["pt"] = (int(event.xdata), int(event.ydata))
            print("Point:", result["pt"])
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.set_title("Click a point")
    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show(block=False)
    plt.waitforbuttonpress(0)
    fig.canvas.mpl_disconnect(cid)
    return result["pt"]


# ------------------------------
# 2. BOX SELECTION (click & drag)
# ------------------------------
def get_box_from_drag(image_np):
    box = {"x1": None, "y1": None, "x2": None, "y2": None}
    dragging = {"active": False}

    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.set_title("Drag to draw box")
    rect = plt.Rectangle((0, 0), 0, 0, fill=False, color="red", linewidth=2)
    ax.add_patch(rect)

    def on_press(event):
        if not event.xdata or not event.ydata:
            return
        dragging["active"] = True
        box["x1"], box["y1"] = int(event.xdata), int(event.ydata)

    def on_move(event):
        if dragging["active"] and event.xdata and event.ydata:
            x2, y2 = int(event.xdata), int(event.ydata)
            rect.set_xy((box["x1"], box["y1"]))
            rect.set_width(x2 - box["x1"])
            rect.set_height(y2 - box["y1"])
            fig.canvas.draw()

    def on_release(event):
        if event.xdata and event.ydata and dragging["active"]:
            box["x2"], box["y2"] = int(event.xdata), int(event.ydata)
            dragging["active"] = False
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plt.show()
    if None in box.values():
        return None
    return [box["x1"], box["y1"], box["x2"], box["y2"]]


# ------------------------------
# 3. LASSO (click multiple points, right-click to finish)
# ------------------------------
def get_polygon_from_clicks(image_np):
    pts = []

    def onclick(event):
        # Left click → add point
        if event.button == 1 and event.xdata and event.ydata:
            pts.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, "ro")
            fig.canvas.draw()

        # Right click → finish
        elif event.button == 3:
            plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.set_title("Left-click to add points, Right-click to finish")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(pts) < 3:
        return None
    return pts



# ------------------------------------------------------------
# MAIN INTERACTION LOOP
# ------------------------------------------------------------
def main():
    image_path = "abc.jpg"
    image_np = np.array(Image.open(image_path).convert("RGB"))

    print("Loading SAM...")
    sam = Segmentation()
    sam.set_image(image_np)

    show_image_with_coordinates(image_np)

    while True:
        print("""
        --------------------------------
                ACTIONS
        --------------------------------
        1 = Point (click)
        2 = Box   (drag)
        3 = Lasso (multi-click)
        4 = Invert Current Mask
        5 = Reset Mask
        0 = Quit
        --------------------------------
        """)

        choice = input("Enter option: ").strip()

        if choice == "0":
            print("Exiting...")
            break

        # POINT (click)
        elif choice == "1":
            pt = get_point_from_click(image_np)
            if pt:
                sam.point_prompt(pt)

        # BOX (drag)
        elif choice == "2":
            box = get_box_from_drag(image_np)
            print("Box:", box)
            if box:
                sam.box_prompt(box)

        # LASSO (multi-click)
        elif choice == "3":
            poly = get_polygon_from_clicks(image_np)
            print("Polygon:", poly)
            if poly:
                sam.lasso_prompt(poly)

        elif choice == "4":
            sam.invert_selection()

        elif choice == "5":
            sam.prev_mask = None
            sam.prev_logits = None
            print("Mask reset.")

        # VISUALIZATION
        if sam.prev_mask is not None:
            vis = visualize_with_mask(image_np, sam.prev_mask)
            show_image_with_coordinates(vis)
        else:
            show_image_with_coordinates(image_np)


if __name__ == "__main__":
    main()
