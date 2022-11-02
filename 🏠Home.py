import streamlit as st

st.set_page_config(page_title="The Doctor", page_icon="🏠")
st.markdown("# The Doctor")

st.image("logo.png")
st.write(
    "Welcome! The Doctor is a univeral medical diagnostic tool with a range of functionality."
)
st.write(
    "Choose a page to the left most consistent with the particular problem you are having, and you can find more information on the individual screens. On the final page, we can help you locate the nearest hospital in case you need to seek medical attention."
)

footer = """
<style>
footer{
    visibility:visible;
}
footer:before{
    content:"Please keep in mind that this app uses predictors based on machine learning algorithms. Although the results are highly accurate, false positive or negative results can occur. If you still have concerns after consulting our app, please contact your doctor or find a hospital using our locator tool.";
    display:block;
    position:relative;
}
</style>
"""

st.markdown(footer, unsafe_allow_html=True)
