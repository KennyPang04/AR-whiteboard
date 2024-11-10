FROM python:3.8.2

ENV HOME /root
WORKDIR /root
COPY . .

RUN pip3 install -r requirements.txt
RUN pip install requests

EXPOSE 8080
CMD python3 -u app.py