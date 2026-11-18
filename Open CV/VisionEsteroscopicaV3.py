import cv2
import numpy as np
from skimage.segmentation import watershed
def posicion(cx0, cx1, cy0, Rex, Rey, l, altura_camara):
    """Función matemática para calcular la posición 3D (X, Y, Z)."""
    A = (cx0 + cx1) * 0.903225 * (1 / Rex)
    B = 0.903227
    
    if (B - A) == 0:
        return [0, 0, 0]
        
    pz = (B * l*1.1071499) / (B - A)
    py = altura_camara + 0.6 * (pz / Rey) * (cy0 - 240)
    px = (cx0) * 0.903225 *(l*1.1071499+(A * l*1.1071499 / (B - A))) *(1 / Rex) - 0.451611668 * (A * l*1.1071499 / (B - A))
    
    return [px, py, pz]

def obtener_deteccion_roja_mejorada(frame, margen=5):
    """Detecta tomates separando oclusiones (frutas que se tocan) mediante Watershed."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    alto_frame, ancho_frame = frame.shape[:2]
    
    # 1. Máscara de color
    # Rango para el rojo puro (tonos bajos de Hue)
    rango_bajo1 = np.array([0, 150, 100])
    rango_alto1 = np.array([8, 255, 255])

    # Rango para el rojo puro (tonos altos de Hue)
    rango_bajo2 = np.array([172, 150, 100])
    rango_alto2 = np.array([180, 255, 255])
    mascara = cv2.inRange(hsv, rango_bajo1, rango_alto1) | cv2.inRange(hsv, rango_bajo2, rango_alto2)
    # Cierre morfológico grande para tapar los huecos causados por reflejos de luz
    kernel_brillo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_brillo)
    # 2. Limpieza Morfológica (Apertura y Cierre)
    ee_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, ee_cierre)
    
# 3. Transformada de distancia y búsqueda de semillas (centros de los tomates)
    dist = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    
    # SOLUCIÓN 1: Suavizar la transformada de distancia. 
    # Esto fusiona los múltiples picos de un tomate alargado en un solo pico central.
    dist_blur = cv2.GaussianBlur(dist, (25, 25), 0)
    
    # SOLUCIÓN 2: Búsqueda de picos locales (Local Maxima)
    # Define la distancia mínima en píxeles que esperas que haya entre el centro de un tomate y otro.
    distancia_minima_centros = 45 
    kernel_picos = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distancia_minima_centros, distancia_minima_centros))
    
    # Dilatamos para encontrar el valor máximo en cada vecindario
    dist_dilatada = cv2.dilate(dist_blur, kernel_picos)
    
    # Un píxel es considerado el centro de un tomate SÓLO si es el punto más alto de su zona
    # y además tiene un grosor mínimo (dist_blur > 15 píxeles de "profundidad")
    semillas = ((dist_blur == dist_dilatada) & (dist_blur > 15)).astype(np.uint8)
    
    # Engrosamos un poco los puntos para que connectedComponents los lea bien
    semillas = cv2.dilate(semillas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, marcadores = cv2.connectedComponents(semillas)
    
    # 4. Watershed: Separar los tomates unidos
    etiquetas = watershed(-dist, marcadores, mask=mascara > 0)
    
    detecciones = []
    
    # 5. Extraer cada tomate por separado
    for etiqueta in range(1, marcadores.max() + 1):
        region = ((etiquetas == etiqueta) & (mascara > 0)).astype(np.uint8)
        contornos, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contornos:
            continue
            
        contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(contorno)
        
        if area > 500:
            x, y, w, h = cv2.boundingRect(contorno)
            
            # REGLA: Visión total del tomate
            if (x > margen) and (y > margen) and ((x + w) < (ancho_frame - margen)) and ((y + h) < (alto_frame - margen)):
                
                # Centroide del círculo mínimo para ignorar partes tapadas
                (cx_float, cy_float), radio = cv2.minEnclosingCircle(contorno)
                cx, cy = int(cx_float), int(cy_float)
                
                detecciones.append({'cx': cx, 'cy': cy, 'bbox': (x, y, x + w, y + h)})
                
    return detecciones

def main():
    cap0 = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
    cap1 = cv2.VideoCapture(2, cv2.CAP_DSHOW) 

    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: No se pudo acceder a una o ambas cámaras.") 
        return

    ancho_frame = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    Rex, Rey = float(ancho_frame), 480.0
    l, altura_camara = 7.7, 10
    
    # Tolerancia en píxeles de diferencia en el eje Y para considerar que es el mismo tomate
    tolerancia_y = 30

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # Se obtienen listas de detecciones
        det_izq = obtener_deteccion_roja_mejorada(frame0)
        det_der = obtener_deteccion_roja_mejorada(frame1)
        
        tomates_emparejados = []
        detecciones_der_disponibles = det_der.copy()
        
        # ASIGNACIÓN: Emparejar usando la coordenada Y (Epipolar geometry)
        for d_izq in det_izq:
            mejor_coincidencia = None
            menor_dif_y = tolerancia_y
            
            for d_der in detecciones_der_disponibles:
                dif_y = abs(d_izq['cy'] - d_der['cy'])
                if dif_y < menor_dif_y:
                    menor_dif_y = dif_y
                    mejor_coincidencia = d_der
            
            if mejor_coincidencia:
                tomates_emparejados.append((d_izq, mejor_coincidencia))
                # Remover para no asignar el mismo tomate derecho a dos izquierdos
                detecciones_der_disponibles.remove(mejor_coincidencia)

        # Procesar e imprimir los tomates emparejados exitosamente
        for idx, (tomate_izq, tomate_der) in enumerate(tomates_emparejados):
            cx0_original = tomate_izq['cx']
            cy0 = tomate_izq['cy']
            cx1_original = tomate_der['cx']
            
            cx0 = ancho_frame - cx0_original 
            cx1 = cx1_original            
            
            px, py, pz = posicion(cx0, cx1, 480-cy0, Rex, Rey, l, altura_camara)
            texto_3d = f"T{idx+1} | PX:{px:.1f} PY:{py:.1f} PZ:{pz:.1f}"

            # --- Dibujar en la Cámara Izquierda ---
            x1_i, y1_i, x2_i, y2_i = tomate_izq['bbox']
            cv2.rectangle(frame0, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            cv2.circle(frame0, (cx0_original, cy0), 5, (0, 0, 255), -1)
            cv2.putText(frame0, texto_3d, (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # --- Dibujar en la Cámara Derecha ---
            x1_d, y1_d, x2_d, y2_d = tomate_der['bbox']
            cv2.rectangle(frame1, (x1_d, y1_d), (x2_d, y2_d), (0, 255, 0), 2)
            cv2.circle(frame1, (cx1_original, tomate_der['cy']), 5, (0, 0, 255), -1)
            cv2.putText(frame1, texto_3d, (x1_d, y1_d - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        combined_frame = cv2.hconcat([frame0, frame1])
        cv2.imshow("Vision Estereoscopica V3", combined_frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 32]:
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()