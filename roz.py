import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json 

folder_path = 'train_1'
shape_y = 2976 // 4
shape_x = 3968 // 4

filename_list = os.listdir(folder_path)
idx = 0
contour_images=[]
sorted_points_list=[]
templates_path = "characters"
templates = {}
sift = cv2.SIFT_create()
orb = cv2.ORB_create()
ransac_reproj_thresh = 5.0

with open('output.json', 'w') as json_file:
    json.dump({}, json_file)

def load_templates(templates_path):
    templates = {}
    for filename in os.listdir(templates_path):
        if filename.endswith(".png"):
            letter = filename.split(".")[0]
            template = cv2.imread(os.path.join(templates_path, filename))
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            ret, th3 = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
            templates[letter] = th3
    return templates

templates = load_templates(templates_path)

def first_mask():
    lower_threshold_b = 109
    upper_threshold_b = 255
    lower_threshold_g = 132
    upper_threshold_g = 255
    lower_threshold_r = 143
    upper_threshold_r = 255
    r, g, b = cv2.split(img)
    _, th_b = cv2.threshold(b, lower_threshold_b, upper_threshold_b, cv2.THRESH_BINARY)
    _, th_g = cv2.threshold(g, lower_threshold_g, upper_threshold_g, cv2.THRESH_BINARY)
    _, th_r = cv2.threshold(r, lower_threshold_r, upper_threshold_r, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(th_r, cv2.bitwise_or(th_g, th_b))
    return combined_mask



while idx < len(filename_list):
    final_results = {}
    img = cv2.imread(os.path.join(folder_path, filename_list[idx]))
    img = cv2.resize(img, (shape_x, shape_y))

    result_img = cv2.bitwise_and(img, img, mask=first_mask())

    gray_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

    ret, th1 = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(opening, 100, 255)
    kernel = np.ones((6,6), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 40000 < area < 200000:
            cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(img, img, mask=mask)

    result[mask == 0] = [0, 0, 0]

    gray_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    ret, th2 = cv2.threshold(gray_img, 150, 200, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(th2,kernel,iterations = 1)

    edges = cv2.Canny(erosion, 100, 255)
    kernel = np.ones((6,6), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 30000 < area:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y-10:y+h+10, x-10:x+w+10]
            scaled_roi = cv2.resize(roi, (1500, 900))

    gray_img = cv2.cvtColor(scaled_roi, cv2.COLOR_BGR2GRAY)
    ret,th3 = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)

    kernel = np.ones((8,8), np.uint8)
    erosion = cv2.erode(th3,kernel,iterations = 1)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.Canny(erosion, 100, 255)

    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for point in sorted_points_list:
        x, y = point
        if point == left_bottom or point == right_bottom:
            y += 10
        elif point == left_top or point == right_top:
            y -= 10
        if point == right_bottom or point == right_top:
            x += 10
        elif point == left_bottom or point == left_top:
            x -= 10
        point[0] = x
        point[1] = y
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    #cv2.drawContours(scaled_roi, [max_contour], 0, (255,255,0), thickness=3)
    if max_contour is not None:
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        points_list = []
        if len(approx) == 4:
            points = approx.reshape(-1, 2)
            for point in points:
                x, y = point
                #cv2.circle(scaled_roi, (x, y), 5, (255), -1)
                points_list.append([x, y])

        distances = []
        high_points = []
        h_p =1000
        if len(approx) > 4:
            points = approx.reshape(-1, 2)
            for point in points:
                x, y = point
                if y < h_p:
                    h_p = y
                    high_points = [x, y]
                h_x , h_y = high_points

            #cv2.circle(scaled_roi, (h_x, h_y), 5, (255), -1)
            points_list.append([h_x, h_y+10])

            for point in points:
                x, y = point
                dis = np.linalg.norm(np.array([h_x, h_y]) - np.array([x, y]))
                distances.append((dis , x ,y))
            distances.sort(key=lambda x: x[0], reverse=True)

            max_distances = distances[:3]
            add_di = 0
            for dis, x, y in max_distances:
                if x != 0 and y != 0 and add_di < 2:
                    add_di += 1
                    #cv2.circle(scaled_roi, (x, y), 5, (255), -1)
                    points_list.append([x, y])

            sorted_points = sorted(points_list, key=lambda point: point[1])
            lowest_x1, lowest_x2 = sorted_points[:2]
            n_x ,n_y = lowest_x2

            distances = []
            for point in points_list:
                x, y = point
                dis = np.linalg.norm(np.array([n_x, n_y]) - np.array([x, y]))
                distances.append((dis , x ,y))
            distances.sort(key=lambda x: x[0], reverse=True)
            disq ,xq,yq = distances[0]
            disc,xc,yc = distances[1]

            for point in points:
                x, y = point
                dis1 = np.linalg.norm(np.array([xq, yq]) - np.array([x, y]))
                dis2 = np.linalg.norm(np.array([xc, yc]) - np.array([x, y]))
                dis = dis1 + dis2
                distances.append((dis , x ,y))
            distances.sort(key=lambda x: x[0], reverse=True)
            last_points = distances[1]
            dis ,x,y = last_points
            #cv2.circle(scaled_roi, (x, y), 5, (255), -1)
            points_list.append([x, y])
           

    left_top = min(points_list, key=lambda x: x[0] + x[1])
    left_bottom = min([point for point in points_list if point not in [left_top]], key=lambda x: x[0] - x[1])
    right_bottom = max(points_list, key=lambda x: x[0] + x[1])
    right_top = max([point for point in points_list if point not in [right_bottom]], key=lambda x: x[0] - x[1])


    sorted_points_list = [left_top, right_top, right_bottom, left_bottom]

    for point in sorted_points_list:
        x, y = point

        if point == left_bottom or point == right_bottom:
            y += 10

        elif point == left_top or point == right_top:
            y -= 10

        if point == right_bottom or point == right_top:
            x += 10

        elif point == left_bottom or point == left_top:
            x -= 10

        point[0] = x
        point[1] = y

                  
    output_pts = np.float32([[0, 0], [1500, 0], [1500, 850], [0, 850]])   

    M = cv2.getPerspectiveTransform(np.float32(sorted_points_list), output_pts)

    transformed_roi = cv2.warpPerspective(scaled_roi, M, (scaled_roi.shape[1], scaled_roi.shape[0]), flags=cv2.INTER_LINEAR)
    gray_img = cv2.cvtColor(transformed_roi, cv2.COLOR_BGR2GRAY)
    ret,th3 = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
    kernel = np.ones((15,15), np.uint8)
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(closing,kernel,iterations = 1)

    edges = cv2.Canny(dilation, 100, 255)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = []
    copy_transformed_roi = transformed_roi.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            #cv2.drawContours(transformed_roi, [contour], 0, (255), thickness=5)
            x,y,w,h = cv2.boundingRect(contour)
            if h >300:
                cv2.rectangle(copy_transformed_roi,(x,y),(x+w,y+h),(0,255,0),8)
                rect.append([x,y,w,h])

    rect_sorted = sorted(rect, key=lambda r: r[0])
    tab = []

    for i, (x, y, w, h) in enumerate(rect_sorted):
        roi = transformed_roi[y:y+h, x:x+w]
        gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, th3 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        #cv2.imshow(f"Match {i}", th3)

        best_match = None
        best_score = -1

        for letter, template in templates.items():
            if len(template.shape) == 3:
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                gray_template = template

            binary_template = np.where(gray_template > 0, 255, 0).astype(np.uint8)
            
            resized_template = cv2.resize(binary_template, (th3.shape[1], th3.shape[0]))

            result = cv2.matchTemplate(th3, resized_template, cv2.TM_CCOEFF)
            max_score = cv2.minMaxLoc(result)[1]

            if max_score > best_score:
                best_score = max_score
                best_match = letter

        #print(f"Najbardziej podobny szablon dla regionu {i}: {best_match}")
        tab.append(best_match)

    with open('output.json', 'r') as json_file:
        final_results = json.load(json_file)

    name = filename_list[idx].split('.')[0]
    result_string = ''.join(tab)
    print(result_string)
    final_results[name] = result_string

    with open('output.json', 'w') as json_file:
            json.dump(final_results, json_file)

    # Wyświetl przekształcony obszar w nowym oknie
    #cv2.imshow('Transformed ROI', transformed_roi)
    #cv2.imshow('Bounding boxes', copy_transformed_roi)
    #cv2.waitKey(0)  
    # cv2.imshow('result', scaled_roi)
    # cv2.waitKey(0)

    idx += 1

cv2.destroyAllWindows