import { ApiService } from './../api.service';
import { Component, OnInit, getPlatform } from '@angular/core';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css'],
  providers: [ApiService]
})
export class DashboardComponent implements OnInit {
  data: any;

  constructor(private api: ApiService) {
    this.data = [{name: 'yo'}, {name: 'fg'}];
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

  ngOnInit() {
  }

}
