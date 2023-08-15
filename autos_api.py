from flask import Flask,render_template,request,redirect
import json
import numpy  as np
import pandas as pd
import pickle


app=Flask(__name__)

@app.route("/")
def start():
    return render_template("autos.html")

@app.route("/getprice", methods=["POST","GET"])
def call():
    data=request.form
    symboling=data["symb"]
    normalized_losses=data["norm"]
    make=data["mke"]
    fuel_type=data["fuel"]
    aspiration=data["asp"]
    num_of_doors=data["doors"]
    body_style=data["style"]
    drive_wheels=data["drivewheel"]
    engine_location=data["engineloc"]
    wheel_base=data["wbase"]
    length=data["len"]
    width=data["wdth"]
    height=data["height"]
    curb_weight=data["wgt"]
    engine_type=data["engine"]
    num_of_cylinders=data["cyl"]
    engine_size=data["engsize"]
    fuel_system=data["fsys"]
    bore=data["bore"]
    stroke=data["strk"]
    compression_ratio=data["c-ratio"]
    horsepower=data["power"]
    peak_rpm=data["rpm"]
    city_mpg=data["cmpg"]
    highway_mpg=data["hmpg"]

    with open("firstmodel.pkl","rb") as file:
        model=pickle.load(file)
    with open("columns.json","r") as file:
        columns_name=json.load(file)
        columns_name["columns"]
    user_input=np.zeros(len(columns_name["columns"]))
    arr_col=np.array(columns_name["columns"])
    with open("Encoded_data.json","r") as file:
        encoaded_data=json.load(file)
    user_input[0]=symboling
    user_input[1]=normalized_losses 
    user_input[2]=fuel_type    
    user_input[3]=aspiration       
    user_input[4]=num_of_doors  
    user_input[5]=engine_location  
    user_input[6]=wheel_base
    user_input[7]=length
    user_input[8]=width
    user_input[9]=height
    user_input[10]=curb_weight
    user_input[11]=num_of_cylinders
    user_input[12]=engine_size
    user_input[13]=bore
    user_input[14]=stroke
    user_input[15]=compression_ratio
    user_input[16]=horsepower
    user_input[17]=peak_rpm
    user_input[18]=city_mpg
    user_input[19]=highway_mpg   

    make_search_string="make_"+make                                    # for one hot encoaded data
    index1=np.where(arr_col==make_search_string)[0][0]
    user_input[index1]=1

    body_style_string ="body-style_"+body_style
    bs_index = np.where(arr_col== body_style_string)[0][0]
    user_input[bs_index] = 1

    drive_wheels_search_string="dw_"+drive_wheels             # for one hot encoaded data
    index=np.where(arr_col==drive_wheels_search_string)[0][0]
    user_input[index]=1

    engine_type_search_string="engine-type_"+engine_type              
    index=np.where(arr_col==engine_type_search_string)[0][0]
    user_input[index]=1

    fuel_system_style_search_string="fuel-system_"+fuel_system
    index=np.where(arr_col==fuel_system_style_search_string)[0][0]
    user_input[index]=1

    inputdata=user_input
    price=model.predict([inputdata])
    print(f"predicted price={price}")

    return str(price)   

if __name__=="__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)