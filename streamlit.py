import streamlit as st
st.title("DMDM : Deepfake Mask Detection Machine")
input_user_name = st.text_input(label="User Name", value="DMDM")
if st.button("Confirm"):
    con = st.container()
    con.caption("Result")
    con.write(f"Hello~ {str(input_user_name)}")