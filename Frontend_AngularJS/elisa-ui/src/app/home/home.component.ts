import { Component, OnInit } from '@angular/core';
import { ApiService } from './../api.service';


@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})

export class HomeComponent {
  home_icons = [{src: '/assets/images/home_icons/angularJs.png'},
                {src: '/assets/images/home_icons/colab.png'},
                {src: '/assets/images/home_icons/python3.png'},
                {src: '/assets/images/home_icons/vscode.png'},
                {src: '/assets/images/home_icons/hadoop.png'},
                {src: '/assets/images/home_icons/github.png'},
                {src: '/assets/images/home_icons/django.png'},
                {src: '/assets/images/home_icons/jupyter.png'},
                ] ;
  logo = '/assets/images/logo.png';
  data: any;

  constructor(private api: ApiService) {
    this.data = [{}];
    this.getPlatform();
  }

  getPlatform = () => {
    this.api.getAllPlatform().subscribe(
      response => {
        this.data = response;
      },
      error => {
        console.log(error);
      }
    );
  }
}
