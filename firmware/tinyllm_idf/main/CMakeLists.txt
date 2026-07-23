# engine 與 model_data.h 的單一來源在 Arduino sketch 資料夾——兩條路線共用同一份程式碼
idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "." "../../tinyllm_arduino")
