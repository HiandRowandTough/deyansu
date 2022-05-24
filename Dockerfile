FROM tensorflow/tensorflow
RUN pip install flask janome cors
COPY app.py /
COPY ./w2i.pickle /
COPY ./weights.hdf5 /
# COPY ./checkpoints /checkpoints
COPY ./userdic.csv /
WORKDIR /
RUN chmod +x /app.py
EXPOSE 5000
CMD [ "python", "/app.py" ]
