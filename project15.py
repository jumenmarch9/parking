def extract_text_from_license_plate(license_plate_image):
    """한국어 우선 번호판 텍스트 추출"""
    global ocr_debug_info
    
    try:
        print("License plate OCR extraction started")
        
        processed = enhanced_preprocessing(license_plate_image)
        
        timestamp = int(time())
        cv2.imwrite(f'/tmp/license_original_{timestamp}.jpg', license_plate_image)
        cv2.imwrite(f'/tmp/license_processed_{timestamp}.jpg', processed)
        
        # 1단계: 한국어 우선 시도
        korean_configs = [
            ('--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후', 'kor+eng'),
            ('--oem 1 --psm 7', 'kor+eng'),
            ('--oem 1 --psm 6', 'kor+eng'),
        ]
        
        for config, lang in korean_configs:
            try:
                text = pytesseract.image_to_string(processed, config=config, lang=lang)
                text_clean = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
                
                if len(text_clean) >= 4:
                    # 한국 번호판 패턴 검증
                    korean_patterns = [
                        r'^[0-9]{2,3}[가-힣][0-9]{4}$',
                        r'^[가-힣]{2}[0-9]{2}[가-힣][0-9]{4}$',
                        r'^[0-9]{2}[가-힣][0-9]{4}$'
                    ]
                    
                    for pattern in korean_patterns:
                        if re.match(pattern, text_clean):
                            print(f"OCR Success (Korean): {text_clean}")
                            ocr_debug_info = f"Success (Korean): {text_clean}"
                            return text_clean
                            
            except Exception as e:
                continue
        
        # 2단계: 한국어 인식 실패 시에만 영어 시도
        english_configs = [
            ('--oem 1 --psm 8', 'eng'),
            ('--oem 3 --psm 8', 'eng')
        ]
        
        for config, lang in english_configs:
            try:
                text = pytesseract.image_to_string(processed, config=config, lang=lang)
                text_clean = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
                
                if len(text_clean) >= 4 and re.match(r'^[A-Z0-9]+$', text_clean):
                    print(f"OCR Success (English): {text_clean}")
                    ocr_debug_info = f"Success (English): {text_clean}"
                    return text_clean
                    
            except Exception as e:
                continue
        
        print("OCR failed - No valid text detected")
        ocr_debug_info = "OCR failed"
        return None
        
    except Exception as e:
        print(f"OCR error: {e}")
        ocr_debug_info = f"OCR error: {e}"
        return None
