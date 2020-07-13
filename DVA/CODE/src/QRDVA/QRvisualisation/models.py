from django.db import models

# Create your models here.
class FileInput(models.Model):
    # set user input output_name as primary key and make sure it is unique
    output_name = models.CharField(max_length = 256, primary_key = False, unique = False)
    TRC_file = models.FileField(upload_to = 'documents/')
    skiprows = models.CharField(max_length = 256, primary_key=False, unique = False)
    #amendment = models.FileField(upload_to = 'amendments/')
    uploaded_at = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return self.output_name
    # create absoulte url to be redirected to
    def get_absolute_url(self):
        return reverse("QRvisualisation:outputview", kwargs = {'pk':self.pk})


class qrOutput(models.Model):
    doc_name = models.ForeignKey('FileInput', on_delete= models.CASCADE)
    TRC_file = models.TextField()
    output1 = models.TextField()
    output2 = models.TextField()

    def __str__(self):
        return str(self.doc_name)


class MaintenanceZonalFileInput(models.Model):
    LINE_CHOICES = (
        ('C138','C138'),
        ('C195','C195'),
    )

    num_quarters = models.CharField(max_length = 256, primary_key=False, unique = False)
    line = models.CharField(max_length = 4, choices=LINE_CHOICES,default='C138')
